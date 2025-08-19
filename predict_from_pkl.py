# predict_from_pkl.py
from pathlib import Path
import argparse, pickle, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset_loader import build_node_features  # uses scalar "type" -> D=10
import json, argparse

NUM_CLASSES = 25
FEAT_NAMES_DEFAULT = [
    'chamfer','through_hole','triangular_passage','rectangular_passage','6sides_passage',
    'triangular_through_slot','rectangular_through_slot','circular_through_slot',
    'rectangular_through_step','2sides_through_step','slanted_through_step','Oring','blind_hole',
    'triangular_pocket','rectangular_pocket','6sides_pocket','circular_end_pocket',
    'rectangular_blind_slot','v_circular_end_blind_slot','h_circular_end_blind_slot',
    'triangular_blind_step','circular_blind_step','rectangular_blind_step','round','stock'
]

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        return self.lin(x)

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim      = int(ckpt.get("in_dim", 10))
    hidden      = int(ckpt.get("hidden", 128))
    num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
    feat_names  = ckpt.get("feat_names", FEAT_NAMES_DEFAULT)
    model = GCN(in_dim, hidden=hidden, out_dim=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, feat_names

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
            json.dump(
                {"labels_idx": idxs,
                "labels_name": names,
                "num_faces": len(idxs)},
                f, indent=2
            )
        print(f"wrote {args.out_json}")

if __name__ == "__main__":
    main()
