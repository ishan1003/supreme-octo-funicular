# dataset_loader.py
import json, pickle
from pathlib import Path
import numpy as np
import torch
# splits.py
from pathlib import Path
import random

def get_available_ids(root: Path, max_id=10000, strict=True):
    root = Path(root)
    good = []
    for i in range(max_id):
        pkl = root / f"{i}.pkl"
        js  = root / "labels" / f"{i}.json"
        if not (pkl.exists() and js.exists()):
            continue
        if strict:
            try:
                with open(pkl, "rb") as f: G = pickle.load(f)
                with open(js, "r") as f:  J = json.load(f)
                if int(G["num_nodes"]) != len(J["per_face_labels"]):
                    # optional: print(f"[skip {i}] nodes={int(G['num_nodes'])} labels={len(J['per_face_labels'])}")
                    continue
            except Exception:
                continue
        good.append(i)
    return good

def split_ids(ids, seed=13, frac=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    ids = np.array(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(frac[0]*n)
    n_va = int(frac[1]*n)
    train = ids[:n_tr].tolist()
    val   = ids[n_tr:n_tr+n_va].tolist()
    test  = ids[n_tr+n_va:].tolist()
    return train, val, test

def _one_hot(idx, n):
    v = np.zeros(n, dtype=np.float32); v[int(idx)] = 1.0; return v

def build_node_features(face_feats: dict, use_type_onehot=False):
    """
    Returns float32 [num_nodes, D] matrix from face_features dict.
    Included: area, adj(deg), loops, centroid(3), convexity(one-hot 3), (optional) type(one-hot)
    """
    n = len(face_feats['area'])
    area      = np.asarray(face_feats['area'], dtype=np.float32).reshape(n,1)
    deg       = np.asarray(face_feats['adj'], dtype=np.float32).reshape(n,1)
    loops     = np.asarray(face_feats['loops'], dtype=np.float32).reshape(n,1)
    centroid  = np.asarray(face_feats['centroid'], dtype=np.float32)  # [n,3]
    conv      = np.asarray(face_feats['convexity'], dtype=np.int64)
    conv_oh   = np.stack([_one_hot(c, 3) for c in conv], axis=0)      # [n,3]

    parts = [area, deg, loops, centroid, conv_oh]

    if use_type_onehot:
        # surface type id from OCC adaptor; make it one-hot within observed range
        stype = np.asarray(face_feats['type'], dtype=np.int64)
        S = int(stype.max()) + 1
        stype_oh = np.stack([_one_hot(t, S) for t in stype], axis=0)
        parts.append(stype_oh)
    else:
        parts.append(np.asarray(face_feats['type'], dtype=np.float32).reshape(n,1))

    x = np.concatenate(parts, axis=1).astype(np.float32)
    # light normalization (helps)
    x[:,0] = (x[:,0] - x[:,0].mean()) / (x[:,0].std()+1e-6)  # area z-norm
    x[:,3:6] = x[:,3:6] / (np.linalg.norm(x[:,3:6], axis=1, keepdims=True)+1e-6)  # centroid direction-ish
    return x

def load_sample(root: Path, idx: int):
    root = Path(root)
    # Pick the step extension you used
    step_path = (root / f"{idx}.stp")
    if not step_path.exists():
        step_path = root / f"{idx}.step"
    pkl_path  = root / f"{idx}.pkl"
    json_path = root / "labels" / f"{idx}.json"
    if not (pkl_path.exists() and json_path.exists()):
        raise FileNotFoundError(idx, step_path, pkl_path, json_path)

    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    with open(json_path, "r") as f:
        J = json.load(f)

    num_nodes = int(G['num_nodes'])
    y = np.array(J['per_face_labels'], dtype=np.int64)
    assert len(y) == num_nodes, f"len(labels)={len(y)} != num_nodes={num_nodes} (idx={idx})"

    x = build_node_features(G['face_features'])
    edge_index = np.asarray(G['edge_index'], dtype=np.int64)  # shape [2, E]

    # torch tensors for PyG
    data = {
        "x": torch.from_numpy(x),
        "edge_index": torch.from_numpy(edge_index),
        "y": torch.from_numpy(y),
        "idx": idx
    }
    return data