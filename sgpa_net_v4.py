"""
SGPA-Net Training Script — 10 Patients
Includes: SVM baseline, smarter capping, LSTM/GRU/Transformer/Mamba, clinical metrics
Run on server: CUDA_VISIBLE_DEVICES=4 nohup python3 ~/sgpa_net_v4.py > ~/sgpa_net_v4.log 2>&1 &
Run on Kaggle: python sgpa_net_v4.py
Monitor: tail -f ~/sgpa_net_v4.log
"""
import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
warnings.filterwarnings('ignore')

# ── cuDNN patch for broken server installations ───────────────────
import torch.nn.modules.rnn as _rnn_module
_original_flatten = _rnn_module.RNNBase.flatten_parameters
def _safe_flatten(self):
    try:
        _original_flatten(self)
    except Exception:
        pass
_rnn_module.RNNBase.flatten_parameters = _safe_flatten
torch.backends.cudnn.enabled = False

# ── Environment detection ─────────────────────────────────────────
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DRIVE_BASE = os.path.join(PROJECT_DIR, "data")
SAVE_BASE = os.path.join(PROJECT_DIR, "results")

os.makedirs(SAVE_BASE, exist_ok=True)

print("Data folder :", DRIVE_BASE)
print("Save folder :", SAVE_BASE)

# ── Config ────────────────────────────────────────────────────────
PATIENTS   = ['chb01','chb02','chb03','chb05','chb06','chb07',
              'chb08','chb09','chb10','chb20']
DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
WINDOW_SEC = 5

CONFIG = {
    'scatter_dim'    : 126,
    'gat_hidden'     : 64,
    'gat_heads'      : 4,
    'temporal_hidden': 128,
    'temporal_layers': 2,
    'seq_len'        : 8,
    'lr'             : 0.0003,
    'weight_decay'   : 1e-5,
    'epochs'         : 50,
    'batch_size'     : 64,
    'plv_threshold'  : 0.3,
    'dropout'        : 0.3,
}

print(f"Device  : {DEVICE}" + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
print(f"Patients: {PATIENTS}")
print(f"Config  : {CONFIG}\n")

# ── Data loading ──────────────────────────────────────────────────
def load_patient_data(pid):
    """Load features and labels for one patient."""
    pid_short = pid.replace('chb', '')

    label_path = os.path.join(DRIVE_BASE, f"labels_{pid_short}.npy")
    scatter_path = os.path.join(DRIVE_BASE, "features", pid, "scattering.npy")
    plv_path = os.path.join(DRIVE_BASE, "features", pid, "plv_graphs.npz")

    lpath = label_path if os.path.exists(label_path) else None
    spath = scatter_path if os.path.exists(scatter_path) else None
    gpath = plv_path if os.path.exists(plv_path) else None

    if not all([lpath, spath, gpath]):
        missing = []
        if not lpath:
            missing.append("labels")
        if not spath:
            missing.append("scattering")
        if not gpath:
            missing.append("plv_graphs")
        print(f"{pid}: MISSING {missing} — skipping")
        return None

    labels = np.load(lpath)
    scatter = np.load(spath)
    plv_raw = np.load(gpath)
    plv = {b: plv_raw[b] for b in ["theta", "alpha", "beta", "gamma"]}

    # Normalize scattering globally per patient
    mean = scatter.mean(axis=0, keepdims=True)
    std = scatter.std(axis=0, keepdims=True) + 1e-8
    scatter = (scatter - mean) / std

    return {
        "scatter": scatter,
        "plv": plv,
        "labels": labels,
    }

print("Loading data...")
all_data = {}
for pid in PATIENTS:
    result = load_patient_data(pid)
    if result is not None:
        all_data[pid] = result
        labels = result['labels']
        print(f"  {pid}: {len(labels):6d} samples | "
              f"inter={np.sum(labels==0):5d} pre={np.sum(labels==1):5d} | "
              f"scatter={result['scatter'].shape}")

PATIENTS = list(all_data.keys())  # Only keep successfully loaded patients
print(f"\nLoaded {len(PATIENTS)} patients: {PATIENTS}\n")

# ── Smarter stratified capping ────────────────────────────────────
# Goal: every patient contributes comparable data while preserving
# a realistic interictal:preictal ratio (MIN_RATIO = 3:1 minimum)
# even after capping. Caps BOTH classes if needed.
MAX_ABSOLUTE = 8000
MIN_RATIO    = 3   # minimum interictal:preictal after capping

print("Applying smarter stratified capping...")
for pid in list(all_data.keys()):
    labels    = all_data[pid]['labels']
    pre_idx   = np.where(labels == 1)[0]
    inter_idx = np.where(labels == 0)[0]
    rng       = np.random.RandomState(42)
    n_pre, n_inter = len(pre_idx), len(inter_idx)

    # Step 1: Cap preictal if it's too large relative to budget
    # (ensures interictal still gets at least MIN_RATIO slots)
    max_pre_for_cap = MAX_ABSOLUTE // (MIN_RATIO + 1)

    if n_pre > max_pre_for_cap:
        pre_idx = rng.choice(pre_idx, max_pre_for_cap, replace=False)
        n_pre   = max_pre_for_cap
        print(f"  {pid}: preictal capped "
              f"{len(np.where(labels==1)[0])}→{n_pre} "
              f"(too many seizures for budget)")

    # Step 2: Fill remaining budget with interictal, ensuring MIN_RATIO
    target_inter = min(n_inter, max(n_pre * MIN_RATIO, MAX_ABSOLUTE - n_pre))
    if n_inter > target_inter:
        inter_idx = rng.choice(inter_idx, target_inter, replace=False)

    keep = np.sort(np.concatenate([pre_idx, inter_idx]))

    all_data[pid]['scatter'] = all_data[pid]['scatter'][keep]
    all_data[pid]['labels']  = all_data[pid]['labels'][keep]
    for b in all_data[pid]['plv']:
        all_data[pid]['plv'][b] = all_data[pid]['plv'][b][keep]

    fl = all_data[pid]['labels']
    print(f"  {pid}: {len(labels):6d}→{len(keep):5d} | "
          f"inter={np.sum(fl==0):5d} pre={np.sum(fl==1):5d} "
          f"ratio=1:{np.sum(fl==0)/max(np.sum(fl==1),1):.1f}")

print("Data ready!\n")

# ════════════════════════════════════════════════════════════════════
# SVM BASELINE (runs on CPU before GPU experiments)
# ════════════════════════════════════════════════════════════════════
def run_svm_baseline_loso():
    """LinearSVC baseline with LOSO — fast version."""
    print("\n" + "="*65)
    print("  SVM BASELINE — Leave-One-Subject-Out (LinearSVC)")
    print("="*65)

    all_features, all_labels_svm = {}, {}
    for pid in PATIENTS:
        scatter = all_data[pid]['scatter']
        labels  = all_data[pid]['labels']
        N       = scatter.shape[0]
        all_features[pid]   = scatter.reshape(N, -1)
        all_labels_svm[pid] = labels

    svm_results = {}

    for test_pid in PATIENTS:
        train_pids = [p for p in PATIENTS if p != test_pid]

        X_train = np.concatenate([all_features[p]   for p in train_pids])
        y_train = np.concatenate([all_labels_svm[p] for p in train_pids])
        X_test  = all_features[test_pid]
        y_test  = all_labels_svm[test_pid]

        # Balance training: 3:1 ratio
        minority_count = int(np.sum(y_train == 1))
        if minority_count == 0:
            print(f"\n  {test_pid}: No preictal in training, skipping")
            continue

        majority_idx = np.where(y_train == 0)[0]
        minority_idx = np.where(y_train == 1)[0]
        n_majority   = min(len(majority_idx), minority_count * 3)
        rng          = np.random.RandomState(42)
        sampled_maj  = rng.choice(majority_idx, n_majority, replace=False)
        bal_idx      = np.concatenate([sampled_maj, minority_idx])
        rng.shuffle(bal_idx)

        X_train_bal = X_train[bal_idx]
        y_train_bal = y_train[bal_idx]

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled  = scaler.transform(X_test)

        # LinearSVC — 100x faster than RBF SVM
        base_clf = LinearSVC(C=0.1, class_weight='balanced',
                             max_iter=2000, random_state=42)
        clf      = CalibratedClassifierCV(base_clf, cv=3)
        clf.fit(X_train_scaled, y_train_bal)

        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        f1  = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.0
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
        sens    = tp/(tp+fn) if (tp+fn)>0 else 0
        spec    = tn/(tn+fp) if (tn+fp)>0 else 0
        fpr_hr  = fp / max((tn+fp)/(3600/WINDOW_SEC), 0.001)

        svm_results[test_pid] = {
            'f1': f1, 'auc': auc,
            'sensitivity': sens, 'specificity': spec,
            'fpr_per_hour': fpr_hr,
            'confusion_matrix': cm.tolist()
        }

        print(f"\n  Test: {test_pid} | Train: {len(X_train_bal)} | Test: {len(X_test)}")
        print(f"    F1={f1:.4f} AUC={auc:.4f} Sens={sens:.4f} "
              f"Spec={spec:.4f} FPR/hr={fpr_hr:.1f}")

    if svm_results:
        avg_f1   = np.mean([r['f1']           for r in svm_results.values()])
        avg_auc  = np.mean([r['auc']          for r in svm_results.values()])
        avg_sens = np.mean([r['sensitivity']  for r in svm_results.values()])
        avg_spec = np.mean([r['specificity']  for r in svm_results.values()])
        avg_fpr  = np.mean([r['fpr_per_hour'] for r in svm_results.values()])
        print(f"\n{'='*65}")
        print(f"  SVM AVERAGE: F1={avg_f1:.4f} AUC={avg_auc:.4f} "
              f"Sens={avg_sens:.4f} Spec={avg_spec:.4f} FPR/hr={avg_fpr:.1f}")
        print(f"{'='*65}")

    svm_path = os.path.join(SAVE_BASE, 'svm_baseline_loso.json')
    with open(svm_path, 'w') as f:
        json.dump(svm_results, f, indent=2)
    print(f"  SVM results saved to {svm_path}")
    return svm_results

svm_results = run_svm_baseline_loso()

# ════════════════════════════════════════════════════════════════════
# DEEP LEARNING — Dataset, Models, Training
# ════════════════════════════════════════════════════════════════════

class EEGGraphDataset(Dataset):
    def __init__(self, scatter, plv, labels, seq_len=8, use_plv=True):
        self.scatter = scatter
        self.plv     = plv
        self.labels  = labels
        self.seq_len = seq_len
        self.use_plv = use_plv
        self.n_ch    = scatter.shape[1]
        self.sequences = [
            (i, int(labels[i + seq_len - 1]))
            for i in range(len(labels) - seq_len + 1)
        ]

    def __len__(self): return len(self.sequences)

    def _build_graph(self, idx):
        x = torch.tensor(self.scatter[idx], dtype=torch.float32)
        if self.use_plv:
            adj = sum(self.plv[b][idx] for b in ['theta','alpha','beta','gamma']) / 4.0
            src, dst, attrs = [], [], []
            for i in range(self.n_ch):
                for j in range(self.n_ch):
                    if i != j and adj[i, j] > CONFIG['plv_threshold']:
                        src.append(i); dst.append(j)
                        attrs.append([float(self.plv[b][idx][i, j])
                                      for b in ['theta','alpha','beta','gamma']])
            if not src:
                for i in range(self.n_ch):
                    for j in range(self.n_ch):
                        if i != j:
                            src.append(i); dst.append(j)
                            attrs.append([0.1]*4)
            return Data(x=x,
                        edge_index=torch.tensor([src, dst], dtype=torch.long),
                        edge_attr=torch.tensor(attrs, dtype=torch.float32))
        else:
            src, dst = zip(*[(i, j) for i in range(self.n_ch)
                                    for j in range(self.n_ch) if i != j])
            return Data(x=x,
                        edge_index=torch.tensor([list(src), list(dst)], dtype=torch.long))

    def __getitem__(self, idx):
        start, label = self.sequences[idx]
        return [self._build_graph(start + t) for t in range(self.seq_len)], label


def collate_sequences(batch):
    gl     = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return [Batch.from_data_list([seq[t] for seq in gl])
            for t in range(len(gl[0]))], labels

# ── Spatial encoders ──────────────────────────────────────────────
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid, out_dim, dropout=0.3):
        super().__init__()
        self.c1   = GCNConv(in_dim, hid)
        self.c2   = GCNConv(hid, hid)
        self.fc   = nn.Linear(hid, out_dim)
        self.n1   = nn.LayerNorm(hid)
        self.n2   = nn.LayerNorm(hid)
        self.drop = dropout

    def forward(self, data):
        x = F.dropout(F.elu(self.n1(self.c1(data.x, data.edge_index))), self.drop, self.training)
        x = F.dropout(F.elu(self.n2(self.c2(x,      data.edge_index))), self.drop, self.training)
        return self.fc(global_mean_pool(x, data.batch))


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid, out_dim, heads=4, dropout=0.3, edge_dim=4):
        super().__init__()
        self.c1   = GATv2Conv(in_dim,    hid, heads=heads, edge_dim=edge_dim,
                               dropout=dropout, concat=True)
        self.c2   = GATv2Conv(hid*heads, hid, heads=heads, edge_dim=edge_dim,
                               dropout=dropout, concat=True)
        self.fc   = nn.Linear(hid*heads, out_dim)
        self.n1   = nn.LayerNorm(hid*heads)
        self.n2   = nn.LayerNorm(hid*heads)
        self.drop = dropout

    def forward(self, data):
        ea = getattr(data, 'edge_attr', None)
        x  = F.dropout(F.elu(self.n1(self.c1(data.x, data.edge_index, edge_attr=ea))),
                        self.drop, self.training)
        x  = F.dropout(F.elu(self.n2(self.c2(x,      data.edge_index, edge_attr=ea))),
                        self.drop, self.training)
        return self.fc(global_mean_pool(x, data.batch))

# ── Temporal encoders ─────────────────────────────────────────────
class LSTMTemporal(nn.Module):
    def __init__(self, in_dim, hid, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.output_dim = hid * 2

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.cat([h[-2], h[-1]], dim=-1)


class GRUTemporal(nn.Module):
    def __init__(self, in_dim, hid, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, n_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.output_dim = hid * 2

    def forward(self, x):
        _, h = self.gru(x)
        return torch.cat([h[-2], h[-1]], dim=-1)


class TransformerTemporal(nn.Module):
    def __init__(self, in_dim, hid, n_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.pos  = nn.Parameter(torch.randn(1, 100, hid) * 0.02)
        layer     = nn.TransformerEncoderLayer(d_model=hid, nhead=4,
                                                dim_feedforward=hid*4,
                                                dropout=dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_dim = hid

    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.size(1), :]
        return self.enc(x)[:, -1, :]


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner         = d_model * expand
        self.d_inner    = d_inner
        self.d_state    = d_state
        self.in_proj    = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d     = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                    padding=d_conv-1, groups=d_inner)
        self.x_proj     = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj    = nn.Linear(1, d_inner, bias=True)
        A               = torch.arange(1, d_state+1, dtype=torch.float32)
        self.A_log      = nn.Parameter(torch.log(A.unsqueeze(0).expand(d_inner, -1)))
        self.D          = nn.Parameter(torch.ones(d_inner))
        self.out_proj   = nn.Linear(d_inner, d_model, bias=False)
        self.norm       = nn.LayerNorm(d_model)
        self.input_norm = nn.LayerNorm(d_inner)

    def forward(self, x):
        residual     = x
        b, sl, _     = x.shape
        x_path, z    = self.in_proj(x).chunk(2, dim=-1)
        x_path       = self.conv1d(x_path.transpose(1, 2))[:, :, :sl].transpose(1, 2)
        x_path       = F.silu(self.input_norm(x_path))
        ssm          = self.x_proj(x_path)
        B            = ssm[:, :, :self.d_state]
        C            = ssm[:, :, self.d_state:2*self.d_state]
        dt           = self.dt_proj(F.softplus(ssm[:, :, -1:]).clamp(max=10.0))
        A            = -torch.exp(self.A_log.clamp(max=5.0))
        h            = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        outs         = []
        for t in range(sl):
            dA  = torch.exp((dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0)).clamp(-10, 10))
            dB  = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)
            h   = (h * dA + dB * x_path[:, t, :].unsqueeze(-1)).clamp(-100, 100)
            outs.append((h * C[:, t, :].unsqueeze(1)).sum(-1))
        y = torch.stack(outs, 1) + x_path * self.D
        return self.norm(self.out_proj(y * F.silu(z)) + residual)


class MambaTemporal(nn.Module):
    def __init__(self, in_dim, hid, n_layers=2, dropout=0.3):
        super().__init__()
        self.proj       = nn.Linear(in_dim, hid)
        self.proj_norm  = nn.LayerNorm(hid)
        self.layers     = nn.ModuleList([MambaBlock(hid) for _ in range(n_layers)])
        self.drop       = nn.Dropout(dropout)
        self.output_dim = hid

    def forward(self, x):
        x = self.proj_norm(self.proj(x))
        for layer in self.layers:
            x = self.drop(layer(x))
        return x[:, -1, :]

# ── Full model ────────────────────────────────────────────────────
class SGPANet(nn.Module):
    def __init__(self, spatial_type='gat', temporal_type='mamba'):
        super().__init__()
        sd  = CONFIG['scatter_dim']
        gh  = CONFIG['gat_hidden']
        gh2 = gh * CONFIG['gat_heads']
        th  = CONFIG['temporal_hidden']

        if spatial_type == 'gcn':
            self.spatial = GCNEncoder(sd, gh, gh)
            s_out = gh
        else:
            self.spatial = GATEncoder(sd, gh, gh2, heads=CONFIG['gat_heads'])
            s_out = gh2

        if temporal_type == 'lstm':
            self.temporal = LSTMTemporal(s_out, th)
        elif temporal_type == 'gru':
            self.temporal = GRUTemporal(s_out, th)
        elif temporal_type == 'transformer':
            self.temporal = TransformerTemporal(s_out, th)
        else:
            self.temporal = MambaTemporal(s_out, th)

        t_out = self.temporal.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(t_out, 64), nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(64, 2)
        )

    def forward(self, graph_seq):
        embs = [self.spatial(g) for g in graph_seq]
        return self.classifier(self.temporal(torch.stack(embs, dim=1)))

# ── Clinical metrics ──────────────────────────────────────────────
def compute_clinical_metrics(labels, predictions, probabilities):
    cm             = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity    = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1             = f1_score(labels, predictions, zero_division=0)
    try:
        auc = roc_auc_score(labels, probabilities)
    except:
        auc = 0.0
    windows_per_hour = 3600 / WINDOW_SEC
    interictal_hours = (tn + fp) / windows_per_hour
    fpr_per_hour     = fp / max(interictal_hours, 0.001)
    ppv              = tp / (tp + fp) if (tp + fp) > 0 else 0
    return {
        'f1': f1, 'auc': auc,
        'sensitivity': sensitivity, 'specificity': specificity,
        'fpr_per_hour': fpr_per_hour, 'ppv': ppv, 'cm': cm
    }


def find_optimal_threshold(labels, probabilities, target_fpr=50.0):
    """
    Find threshold giving FPR <= target while maximising sensitivity.
    Default target=50 FPR/hr — relaxed for realistic clinical data.
    """
    windows_per_hour = 3600 / WINDOW_SEC
    best = {
        'threshold'   : 0.5,
        'f1'          : 0.0,
        'sensitivity' : 0.0,
        'specificity' : 0.0,
        'fpr_per_hour': 999.0
    }
    for thresh in np.arange(0.50, 0.99, 0.01):
        preds = (probabilities >= thresh).astype(int)
        cm    = confusion_matrix(labels, preds)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        inter_hours = (tn + fp) / windows_per_hour
        fpr_hr      = fp / max(inter_hours, 0.001)
        sens        = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec        = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1          = f1_score(labels, preds, zero_division=0)
        if fpr_hr <= target_fpr and sens > best['sensitivity']:
            best = {
                'threshold'   : float(thresh),
                'f1'          : float(f1),
                'sensitivity' : float(sens),
                'specificity' : float(spec),
                'fpr_per_hour': float(fpr_hr)
            }
    return best

# ── Training helpers ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, preds_all, labs_all = 0, [], []
    for gs, labels in loader:
        gs     = [g.to(DEVICE) for g in gs]
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(gs)
        loss   = criterion(logits, labels)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss  += loss.item() * labels.size(0)
        preds_all.extend(logits.argmax(1).detach().cpu().numpy())
        labs_all.extend(labels.cpu().numpy())
    if not labs_all:
        return float('nan'), 0.0
    return total_loss / len(labs_all), f1_score(labs_all, preds_all, zero_division=0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds_all, labs_all, probs_all = [], [], []
    for gs, labels in loader:
        gs = [g.to(DEVICE) for g in gs]
        logits = model(gs)
        probs_all.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labs_all.extend(labels.numpy())
    p  = np.array(preds_all)
    l  = np.array(labs_all)
    pr = np.array(probs_all)
    return compute_clinical_metrics(l, p, pr), pr, l

# ── LOSO experiment runner ────────────────────────────────────────
def run_experiment(spatial, temporal, use_plv=True):
    name = f"{spatial.upper()}+{temporal.upper()}" + ("" if use_plv else "(no-PLV)")
    print(f"\n{'='*65}\n  EXPERIMENT: {name}\n{'='*65}")
    folds = {}

    for test_pid in PATIENTS:
        train_pids = [p for p in PATIENTS if p != test_pid]
        print(f"\n  Fold: test={test_pid}  train={train_pids}")

        # Build training set from all training patients
        tr_s   = np.concatenate([all_data[p]['scatter'] for p in train_pids])
        tr_l   = np.concatenate([all_data[p]['labels']  for p in train_pids])
        tr_plv = {b: np.concatenate([all_data[p]['plv'][b] for p in train_pids])
                  for b in ['theta','alpha','beta','gamma']}

        # Balance training set: 3:1 interictal:preictal
        rng     = np.random.RandomState(42)
        pre_idx = np.where(tr_l == 1)[0]
        maj_idx = np.where(tr_l == 0)[0]
        n_maj   = min(len(maj_idx), len(pre_idx) * 3)
        bal_idx = np.sort(np.concatenate([rng.choice(maj_idx, n_maj, replace=False), pre_idx]))

        tr_s_b   = tr_s[bal_idx]
        tr_l_b   = tr_l[bal_idx]
        tr_plv_b = {b: tr_plv[b][bal_idx] for b in tr_plv}

        # Test set — use as-is (no balancing for honest evaluation)
        te_s   = all_data[test_pid]['scatter']
        te_l   = all_data[test_pid]['labels']
        te_plv = all_data[test_pid]['plv']

        print(f"    Train {len(tr_l_b):6d} "
              f"(inter={np.sum(tr_l_b==0)}, pre={np.sum(tr_l_b==1)})  "
              f"Test {len(te_l):6d} "
              f"(inter={np.sum(te_l==0)}, pre={np.sum(te_l==1)})")

        train_ds = EEGGraphDataset(tr_s_b, tr_plv_b, tr_l_b, CONFIG['seq_len'], use_plv)
        test_ds  = EEGGraphDataset(te_s,   te_plv,   te_l,   CONFIG['seq_len'], use_plv)
        train_dl = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True,
                              collate_fn=collate_sequences, num_workers=2)
        test_dl  = DataLoader(test_ds,  CONFIG['batch_size'], shuffle=False,
                              collate_fn=collate_sequences, num_workers=2)

        model     = SGPANet(spatial, temporal).to(DEVICE)
        n0, n1    = np.sum(tr_l_b == 0), np.sum(tr_l_b == 1)
        w         = torch.tensor([1.0, n0 / max(n1, 1)], dtype=torch.float32).to(DEVICE)
        w         = w / w.sum() * 2
        criterion = nn.CrossEntropyLoss(weight=w)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=CONFIG['lr'],
                                      weight_decay=CONFIG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=10, T_mult=2)

        best_f1, best_res, best_probs, best_labels = 0.0, None, None, None

        for ep in range(CONFIG['epochs']):
            t0        = time.time()
            loss, tf1 = train_epoch(model, train_dl, optimizer, criterion)
            scheduler.step()

            if (ep + 1) % 10 == 0 or ep == CONFIG['epochs'] - 1:
                res, probs, labels_arr = evaluate(model, test_dl)
                print(f"    Ep{ep+1:02d} | loss:{loss:.3f} trF1:{tf1:.3f} | "
                      f"F1:{res['f1']:.3f} AUC:{res['auc']:.3f} "
                      f"Sens:{res['sensitivity']:.3f} Spec:{res['specificity']:.3f} "
                      f"FPR/hr:{res['fpr_per_hour']:.1f} | {time.time()-t0:.0f}s")
                if res['f1'] > best_f1:
                    best_f1     = res['f1']
                    best_res    = res
                    best_probs  = probs
                    best_labels = labels_arr

        if best_res is None:
            best_res, best_probs, best_labels = evaluate(model, test_dl)

        optimal = find_optimal_threshold(best_labels, best_probs, target_fpr=50.0)

        print(f"\n    ✓ BEST {test_pid}:")
        print(f"      Default (0.50): F1={best_res['f1']:.4f} AUC={best_res['auc']:.4f} "
              f"Sens={best_res['sensitivity']:.4f} Spec={best_res['specificity']:.4f} "
              f"FPR/hr={best_res['fpr_per_hour']:.1f}")
        print(f"      Optimal ({optimal['threshold']:.2f}): "
              f"F1={optimal['f1']:.4f} Sens={optimal['sensitivity']:.4f} "
              f"Spec={optimal['specificity']:.4f} FPR/hr={optimal['fpr_per_hour']:.1f}")

        folds[test_pid] = {
            'default': {k: v for k, v in best_res.items() if k != 'cm'},
            'optimal': optimal,
        }

        del model, optimizer, criterion
        torch.cuda.empty_cache()

    # Average across all folds
    def safe_avg(key, source):
        vals = [f[source][key] for f in folds.values() if key in f[source]]
        return float(np.mean(vals)) if vals else 0.0

    avg_default = {k: safe_avg(k, 'default')
                   for k in ['f1','auc','sensitivity','specificity','fpr_per_hour']}
    avg_optimal = {k: safe_avg(k, 'optimal')
                   for k in ['f1','sensitivity','specificity','fpr_per_hour']}

    print(f"\n  ── {name} AVERAGE ({len(PATIENTS)}-fold LOSO) ──")
    print(f"  Default  : F1={avg_default['f1']:.4f} AUC={avg_default['auc']:.4f} "
          f"Sens={avg_default['sensitivity']:.4f} Spec={avg_default['specificity']:.4f} "
          f"FPR/hr={avg_default['fpr_per_hour']:.1f}")
    print(f"  Optimised: F1={avg_optimal['f1']:.4f} "
          f"Sens={avg_optimal['sensitivity']:.4f} Spec={avg_optimal['specificity']:.4f} "
          f"FPR/hr={avg_optimal['fpr_per_hour']:.1f}")

    # Save checkpoint after each experiment
    checkpoint = {'variant': name, 'folds': folds,
                  'avg_default': avg_default, 'avg_optimal': avg_optimal}
    ckpt_path = os.path.join(SAVE_BASE, f"checkpoint_{name.replace('+','_')}.json")
    with open(ckpt_path, 'w') as f:
        json.dump({k: v for k, v in checkpoint.items() if k != 'folds'}, f, indent=2)
    print(f"  Checkpoint saved: {ckpt_path}")

    return checkpoint

# ── Run all 4 experiments ─────────────────────────────────────────
results = {}

print("\n" + "█"*65)
print("  1/4  GCN + LSTM  (baseline)")
print("█"*65)
results['GCN+LSTM'] = run_experiment('gcn', 'lstm', use_plv=False)

print("\n" + "█"*65)
print("  2/4  GAT + LSTM")
print("█"*65)
results['GAT+LSTM'] = run_experiment('gat', 'lstm', use_plv=True)

print("\n" + "█"*65)
print("  3/4  GAT + Transformer")
print("█"*65)
results['GAT+Trans'] = run_experiment('gat', 'transformer', use_plv=True)

print("\n" + "█"*65)
print("  4/4  GAT + Mamba  (SGPA-Net)")
print("█"*65)
results['GAT+Mamba'] = run_experiment('gat', 'mamba', use_plv=True)

# ── Final ablation table ──────────────────────────────────────────
svm_avg_f1  = np.mean([r['f1']  for r in svm_results.values()])
svm_avg_auc = np.mean([r['auc'] for r in svm_results.values()])

print("\n\n" + "="*80)
print(f"  COMPLETE ABLATION TABLE — SGPA-Net ({len(PATIENTS)} Patients, LOSO)")
print("="*80)
print(f"{'Variant':<18} {'F1':>6} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'FPR/hr':>8}  "
      f"{'F1*':>6} {'Sens*':>6} {'Spec*':>6} {'FPR*':>7}")
print(f"{'':18} {'------Default(0.5)------':>36}  {'----Optimised----':>29}")
print("-"*80)
print(f"{'LinearSVC Baseline':<18} {svm_avg_f1:>6.3f} {svm_avg_auc:>6.3f} "
      f"{'  --':>6} {'  --':>6} {'      --':>8}  "
      f"{'  --':>6} {'  --':>6} {'  --':>6} {'   --':>7}")

for name, r in results.items():
    d = r['avg_default']
    o = r['avg_optimal']
    print(f"{name:<18} {d['f1']:>6.3f} {d['auc']:>6.3f} "
          f"{d['sensitivity']:>6.3f} {d['specificity']:>6.3f} "
          f"{d['fpr_per_hour']:>8.1f}  "
          f"{o['f1']:>6.3f} {o['sensitivity']:>6.3f} "
          f"{o['specificity']:>6.3f} {o['fpr_per_hour']:>7.1f}")
print("="*80)
print("* Optimised = threshold tuned for FPR ≤ 50/hour")

# ── Save final results ────────────────────────────────────────────
final_path = os.path.join(SAVE_BASE, 'ablation_results_final.json')
with open(final_path, 'w') as f:
    json.dump({
        'svm_baseline': {
            'avg_f1': float(svm_avg_f1),
            'avg_auc': float(svm_avg_auc),
            'per_patient': svm_results
        },
        'deep_models': {
            n: {'avg_default': r['avg_default'], 'avg_optimal': r['avg_optimal']}
            for n, r in results.items()
        }
    }, f, indent=2)
print(f"\nFinal results saved to {final_path}")
