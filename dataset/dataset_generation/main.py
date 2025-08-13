"""Creates dataset from random combination of machining features

Used to generate dataset of stock cube with machining features applied to them.
The number of machining features is defined by the combination range.
To change the parameters of each machining feature, please see parameters.py
"""

from multiprocessing import Pool
from itertools import combinations_with_replacement
import Utils.shape as shape
import random
import os
import pickle
import json            # NEW
import csv             # NEW
from pathlib import Path  # NEW

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.STEPConstruct import stepconstruct_FindEntity
from OCC.Core.TCollection import TCollection_HAsciiString
import Utils.occ_utils as occ_utils
import feature_creation

from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity


def shape_with_fid_to_step(filename, shape, id_map):
    """Save shape to a STEP file format.

    :param filename: Name to save shape as.
    :param shape: Shape to be saved.
    :param id_map: Variable mapping labels to faces in shape.
    :return: None
    """
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

    writer.Write(filename)

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


if __name__ == '__main__':
    # Parameters to be set before use
    shape_dir = 'data'
    num_features = 24
    combo_range = [3, 10]
    num_samples = 10

    if not os.path.exists(shape_dir):
        os.mkdir(shape_dir)

    combos = []
    for num_combo in range(combo_range[0], combo_range[1]):
        combos += list(combinations_with_replacement(range(num_features), num_combo))

    random.shuffle(combos)
    test_combos = combos[:num_samples]

    for count, combo in enumerate(test_combos):
        print(f"{count}: {combo}")
        generate_shape(shape_dir, combo, count)
