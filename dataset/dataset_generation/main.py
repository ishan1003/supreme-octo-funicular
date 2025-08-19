"""Creates dataset from random combination of machining features

Used to generate dataset of stock cube with machining features applied to them.
The number of machining features is defined by the combination range.
To change the parameters of each machining feature, please see parameters.py
"""
import random
import os
import json            # NEW
import csv             # NEW
from pathlib import Path  # NEW

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.STEPConstruct import stepconstruct_FindEntity
from OCC.Core.TCollection import TCollection_HAsciiString
import Utils.occ_utils as occ_utils
from convert_step_to_graph import convert_step_to_graph
import feature_creation

from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity

import random
from pathlib import Path

import time, gc
import multiprocessing as mp

def _make_one(shape_dir, combo, count):
    # resume-safe inside the worker too
    step_path = Path(shape_dir) / f"{count}.step"
    if step_path.exists():
        return (count, "skip")

    generate_shape(shape_dir, combo, count)
    return (count, "ok")


def random_combo(nfeat: int, k: int):
    # nondecreasing tuple to mimic combinations_with_replacement
    return tuple(sorted(random.choices(range(nfeat), k=k)))

def shape_with_fid_to_step(filename, shape, id_map):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)

    finderp = writer.WS().TransferWriter().FinderProcess()
    faces = occ_utils.list_face(shape)
    loc = TopLoc_Location()

    for face in faces:
        item = stepconstruct_FindEntity(finderp, face, loc)
        if item is None:
            print(face)
            continue
        item.SetName(TCollection_HAsciiString(str(id_map[face])))

    step_path = Path(filename)               # <-- make it a Path
    writer.Write(str(step_path))             # OCC wants a string path

    pkl_path = step_path.with_suffix('.pkl') # same name, .pkl
    # If your module exposes a function convert_step_to_graph(in_path, out_path):
    convert_step_to_graph(str(step_path), str(pkl_path))
    del finderp, faces, loc, writer


# NEW: turn the TopoDS_Face->label map into an ordered list aligned with occ_utils.list_face(shape)
def _per_face_labels(shape, label_map, missing_value=-1):
    labels = []
    for f in occ_utils.list_face(shape):
        try:
            labels.append(int(label_map[f]))
        except Exception:
            labels.append(int(missing_value))
    return labels


# NEW: save sidecar JSON and append a row in a dataset CSV
def save_labels(shape_dir, shape_name, combination, shape, label_map):
    labels_dir = Path(shape_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    per_face = _per_face_labels(shape, label_map, missing_value=-1)

    record = {
        "step_file": f"{shape_name}.step",
        "sample_index": int(shape_name),
        "feature_ids_combo": list(map(int, combination)),  # the tuple you print
        "per_face_labels": per_face
        # Optionally add a class_map here if you have one.
    }

    # Per-sample JSON
    with open(labels_dir / f"{shape_name}.json", "w") as f:
        json.dump(record, f, indent=2)

    # Dataset-wide CSV summary (append-only)
    csv_path = Path(shape_dir) / "labels.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "step_file", "feature_ids_combo"])
        if write_header:
            w.writeheader()
        w.writerow({
            "idx": int(shape_name),
            "step_file": f"{shape_name}.step",
            "feature_ids_combo": " ".join(map(str, combination))
        })

def directive(combo, count):
    shape_name = str(count)
    shapes, face_label_map = feature_creation.shape_from_directive(combo)

    return shapes, shape_name, face_label_map


def save_shape(shape, step_path, label_map):
    print(f"Saving: {step_path}")
    shape_with_fid_to_step(step_path, shape, label_map)


def generate_shape(shape_dir, combination, count):
    """Generate num_shapes random shapes in shape_dir
    :param arg: List of [shape directory path, machining feature combo]
    :return: None
    """
    shape_name = str(count)
    shape, label_map = feature_creation.shape_from_directive(combination)
    step_path = os.path.join(shape_dir, shape_name + '.step')
    save_shape(shape, step_path, label_map)

    # NEW: Write labels sidecars
    save_labels(shape_dir, shape_name, combination, shape, label_map)
    del shape, label_map
    gc.collect()


if __name__ == "__main__":
    shape_dir = "data"
    num_features = 24
    combo_range = [3, 5]   # k in [3,4]
    num_samples = 50000

    Path(shape_dir).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    k_lo, k_hi = combo_range[0], combo_range[1] - 1
    print(f"Streaming {num_samples} samples; k in [{k_lo}, {k_hi}]")

    ctx = mp.get_context("spawn")
    # maxtasksperchild=1 keeps memory flat by restarting workers
    with ctx.Pool(processes=max(1, os.cpu_count() - 1), maxtasksperchild=1000) as pool:
        jobs = []
        for count in range(num_samples):
            # quick skip to avoid queuing already-done work
            if (Path(shape_dir) / f"{count}.step").exists():
                print(f"[{count}] exists, skip", flush=True)
                continue

            k = random.randint(k_lo, k_hi)
            combo = random_combo(num_features, k)
            print(f"[{count}] queue combo={combo}", flush=True)

            jobs.append(pool.apply_async(_make_one, (shape_dir, combo, count)))

        # collect results
        for j in jobs:
            try:
                idx, status = j.get()  # no per-task timeout here; add if you really need it
                if status == "ok":
                    print(f"[{idx}] done", flush=True)
                else:
                    print(f"[{idx}] skipped (already existed)", flush=True)
            except Exception as e:
                print(f"[?] ERROR: {e}", flush=True)
        gc.collect()

