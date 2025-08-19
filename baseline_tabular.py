# baseline_tabular.py
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from dataset_loader import get_available_ids, split_ids, load_sample

root = Path("./dataset/dataset_generation/data")
feat_names = ['chamfer','through_hole','triangular_passage','rectangular_passage','6sides_passage',
              'triangular_through_slot','rectangular_through_slot','circular_through_slot',
              'rectangular_through_step','2sides_through_step','slanted_through_step','Oring','blind_hole',
              'triangular_pocket','rectangular_pocket','6sides_pocket','circular_end_pocket',
              'rectangular_blind_slot','v_circular_end_blind_slot','h_circular_end_blind_slot',
              'triangular_blind_step','circular_blind_step','rectangular_blind_step','round','stock']
NUM_CLASSES = len(feat_names)  # 25
avail = get_available_ids(root, max_id=10000)
print(f"Found {len(avail)} usable parts out of 10000.")
if len(avail) < 50:
    raise RuntimeError("Too few usable samples. Generate more or fix failures.")

train_ids, val_ids, test_ids = split_ids(avail, seed=13)

def stack_faces(id_list):
    Xs, ys = [], []
    for i in id_list:
        d = load_sample(root, i)
        Xs.append(d["x"].numpy())
        ys.append(d["y"].numpy())
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
# ---- assemble ----
Xtr, ytr = stack_faces(train_ids)
Xva, yva = stack_faces(val_ids)
Xte, yte = stack_faces(test_ids)
print(f"Train faces: {Xtr.shape[0]}  Val faces: {Xva.shape[0]}  Test faces: {Xte.shape[0]}  (D={Xtr.shape[1]})")

# ---- class balance ----
counts = np.bincount(ytr, minlength=NUM_CLASSES)
w = 1.0 / (counts + 1e-6)
w *= (len(w) / w.sum())
sample_weight_tr = w[ytr]

# ---- train ----
clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, early_stopping=True, validation_fraction=0.1)
clf.fit(Xtr, ytr, sample_weight=sample_weight_tr)

# ---- eval ----
pred = clf.predict(Xte)
print(classification_report(yte, pred, digits=3, target_names=feat_names))

# Confusion matrix (optional)
cm = confusion_matrix(yte, pred, labels=list(range(NUM_CLASSES)))
plt.figure(figsize=(10,8))
plt.imshow(np.log1p(cm), aspect='auto')  # log scale to see small counts
plt.title("Confusion Matrix (log counts)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.colorbar(); plt.tight_layout(); plt.show()

# ---- quick per-part inference helper ----
def predict_part(part_id: int):
    d = load_sample(root, part_id)
    yhat = clf.predict(d["x"].numpy())
    return [feat_names[int(k)] for k in yhat]

# example:
# print(predict_part(test_ids[0])[:10])
