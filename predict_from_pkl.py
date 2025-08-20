# predict_from_pkl.py  (DeepGCN-only)
from pathlib import Path
import argparse, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset_loader import build_node_features
from collections import OrderedDict

NUM_CLASSES = 25
FEAT_NAMES_DEFAULT = [
    'chamfer','through_hole','triangular_passage','rectangular_passage','6sides_passage',
    'triangular_through_slot','rectangular_through_slot','circular_through_slot',
    'rectangular_through_step','2sides_through_step','slanted_through_step','Oring','blind_hole',
    'triangular_pocket','rectangular_pocket','6sides_pocket','circular_end_pocket',
    'rectangular_blind_slot','v_circular_end_blind_slot','h_circular_end_blind_slot',
    'triangular_blind_step','circular_blind_step','rectangular_blind_step','round','stock'
]

# ----------------- Deep model -----------------
class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.3):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.p    = p_drop
        self.res  = (in_ch == out_ch)
    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, p=self.p, training=self.training)
        if self.res:
            out = out + x
        return out

class DeepGCN(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=4, out_dim=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.in_lin = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([GCNBlock(hidden, hidden, p_drop=dropout) for _ in range(int(layers))])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x, edge_index):
        x = F.relu(self.in_lin(x), inplace=True)
        for blk in self.blocks:
            x = blk(x, edge_index)
        return self.head(x)

# ----------------- Utils -----------------
def _safe_torch_load(path: Path):
    # Prefer weights_only=True to avoid the security warning (PyTorch >=2.4)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _strip_prefix(state_dict, prefixes=("_orig_mod.", "module.")):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd

# -------------- Loader -------------------
def load_model(ckpt_path: Path, device: torch.device):
    ckpt = _safe_torch_load(ckpt_path)
    if "state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing 'state_dict'.")

    # Enforce DeepGCN-only
    arch = ckpt.get("arch", None)
    if arch not in (None, "DeepGCN"):
        raise RuntimeError(f"This predictor only supports DeepGCN checkpoints. Found arch={arch!r}.")

    in_dim      = int(ckpt.get("in_dim", 10))
    hidden      = int(ckpt.get("hidden", 400))   # sensible default for deep ckpt
    layers      = int(ckpt.get("layers", 6))
    num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
    dropout     = float(ckpt.get("dropout", 0.3))
    feat_names  = ckpt.get("feat_names", FEAT_NAMES_DEFAULT)

    state = _strip_prefix(ckpt["state_dict"])  # handle torch.compile / DDP prefixes

    model = DeepGCN(in_dim, hidden=hidden, layers=layers,
                    out_dim=num_classes, dropout=dropout).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, feat_names

# -------------- Predict -------------------
@torch.no_grad()
def predict_from_pkl(pkl_path: Path, ckpt_path: Path, device: torch.device, use_amp=True):
    G = pickle.loads(Path(pkl_path).read_bytes())
    x_np  = build_node_features(G["face_features"], use_type_onehot=False)   # [N,10]
    ei_np = np.asarray(G["edge_index"], dtype=np.int64)                      # [2,E]

    x  = torch.from_numpy(x_np).to(device, non_blocking=True)
    ei = torch.from_numpy(ei_np).to(device, non_blocking=True)

    model, names = load_model(ckpt_path, device)

    if device.type == "cuda" and use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x, ei)
    else:
        logits = model(x, ei)

    idxs = logits.argmax(1).tolist()
    return [names[i] for i in idxs], idxs

# -------------- CLI -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="Path to graph pickle produced from STEP")
    ap.add_argument("--ckpt", default="checkpoints/gcn_facecls.pt")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = (torch.device("cuda") if (args.device=="auto" and torch.cuda.is_available())
              else torch.device(args.device if args.device!="auto" else "cpu"))

    names, idxs = predict_from_pkl(Path(args.pkl), Path(args.ckpt), device)
    print(f"faces: {len(idxs)}")
    print(names[:30], "... (total {})".format(len(names)))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"labels_idx": idxs, "labels_name": names, "num_faces": len(idxs)}, f, indent=2)
        print(f"wrote {args.out_json}")

if __name__ == "__main__":
    main()
