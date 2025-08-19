# visualize_step_with_preds.py
# Headless by default: write a colored STEP from predictions.
# Optional --show opens an interactive viewer (requires PySide2/PyQt5).

import json
from pathlib import Path
import argparse

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.XCAFApp import XCAFApp_Application
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf
from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
from OCC.Core.Interface import Interface_Static

from label_colors import LABEL_COLORS, hex_to_rgb01


def qcolor_from_hex(hx: str) -> Quantity_Color:
    r, g, b = hex_to_rgb01(hx)  # 0..1
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)


def load_step_shape(step_path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def write_colored_step(input_step: str, labels_name: list[str], out_step: str):
    """Write a new STEP with per-face colors according to labels_name."""
    shape = load_step_shape(input_step)
    faces = list(TopologyExplorer(shape, ignore_orientation=True).faces())

    # XDE document (shapes + attributes)
    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document(TCollection_ExtendedString("pythonocc-xde"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    # Add the main shape to the doc
    root_lbl = shape_tool.AddShape(shape)

    # Attach per-face colors
    n = min(len(faces), len(labels_name))
    for i in range(n):
        face = faces[i]
        name = labels_name[i]
        hx = LABEL_COLORS.get(name, "#aaaaaa")
        qcol = qcolor_from_hex(hx)

        # Create / get a label for the subshape (face) and set SURF color
        sub_lbl = shape_tool.AddSubShape(root_lbl, face)
        color_tool.SetColor(sub_lbl, qcol, XCAFDoc_ColorSurf)

    # Write an AP214 STEP with colors
    Interface_Static.SetCVal("write.step.schema", "AP214IS")
    writer = STEPCAFControl_Writer()
    writer.SetColorMode(True)
    ok = writer.Transfer(doc)
    if not ok:
        raise RuntimeError("STEPCAFControl: transfer failed")
    if writer.Write(str(out_step)) != 1:
        raise RuntimeError(f"Failed to write STEP: {out_step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, help="Path to source STEP")
    ap.add_argument("--preds", required=True, help="JSON from predict_from_pkl.py (labels_name)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Transparency (only for --show)")
    ap.add_argument("--wire", action="store_true", help="Wireframe overlay (only for --show)")
    ap.add_argument("--write_step", type=str, default="", help="If set, write colored STEP here")
    ap.add_argument("--show", action="store_true", help="Open an interactive viewer (requires Qt backend)")
    args = ap.parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        raise FileNotFoundError(preds_path)

    with open(preds_path, "r") as f:
        P = json.load(f)
    labels = P.get("labels_name") or []

    # Headless export
    if args.write_step:
        out_step = Path(args.write_step)
        out_step.parent.mkdir(parents=True, exist_ok=True)
        write_colored_step(args.step, labels, str(out_step))
        print(f"[OK] Wrote colored STEP -> {out_step}")

    # Optional interactive visualization
    if args.show:
        # Import the viewer only if we actually show a window
        from OCC.Display.SimpleGui import init_display
        from OCC.Core.AIS import AIS_ColoredShape
        from OCC.Core.Graphic3d import Graphic3d_AlphaMode_Blend

        shape = load_step_shape(args.step)
        faces = list(TopologyExplorer(shape, ignore_orientation=True).faces())
        display, start_display, *_ = init_display()
        ais = AIS_ColoredShape(shape)

        n = min(len(faces), len(labels))
        for i in range(n):
            from OCC.Core.AIS import AIS_ColoredShape
            hx = LABEL_COLORS.get(labels[i], "#aaaaaa")
            ais.SetCustomColor(faces[i], qcolor_from_hex(hx))

        display.Context.Display(ais, True)
        if args.wire:
            display.DisplayShape(shape, update=False, color="black", transparency=0.0, wire=True)
        if 0.0 <= args.alpha < 1.0:
            try:
                display.Context.SetTransparency(ais, 1.0 - args.alpha, True)
            except Exception:
                pass
        display.View_Iso()
        display.FitAll()
        start_display()


if __name__ == "__main__":
    main()
