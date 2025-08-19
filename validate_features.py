# validate_features.py
import pickle, numpy as np
from pathlib import Path

def feature_dim(pkl):
    g = pickle.loads(Path(pkl).read_bytes())
    ff = g["face_features"]
    n = len(ff["area"])
    # scalar features: area(1), deg(1), loops(1), centroid(3), convexity one-hot(3)=9
    base = 1+1+1+3+3
    stype = np.asarray(ff["type"])
    # if you one-hot per-sample (bad), this will vary: stype_dim = stype.max()+1
    # if you use scalar type (good), set stype_dim=1
    stype_dim = 1  # <- set to global dim if you truly one-hot globally
    return base + stype_dim

root = Path("./dataset/dataset_generation/data")
dims = {}
for i in range(20285):
    p = root/f"{i}.pkl"
    j = root/"labels"/f"{i}.json"
    if p.exists() and j.exists():
        d = feature_dim(p)
        dims.setdefault(d, []).append(i)

print("feature dims encountered:", {k: len(v) for k,v in dims.items()})
if len(dims) > 1:
    print("WARNING: inconsistent feature width across parts. Offending groups:", dims)
