#!/usr/bin/env python3
"""
export_to_trt.py
================
Wandelt ONNX‑Modelle in TensorRT‑Engines um.

Beispiel:
    python export_to_trt.py dr.onnx grayscale.onnx             # FP32
    python export_to_trt.py --fp16 *.onnx                      # FP16, falls Nano das kann
"""

# ────────────────────────────────────────────────────────────────────────────────
# Patch für NumPy ≥1.20, damit pycuda.autoinit nicht nach np.bool sucht:
import numpy as np
np.bool = bool
# ────────────────────────────────────────────────────────────────────────────────

import os
import argparse
import tensorrt as trt

# PyCUDA initialisiert direkt die GPU‑Runtime
import pycuda.driver as cuda   # noqa: F401
import pycuda.autoinit         # noqa: F401

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path: str, fp16: bool) -> trt.ICudaEngine:
    """Parst eine ONNX‑Datei und baut eine TensorRT‑Engine."""
    # EXPLICIT_BATCH -> moderne ONNX‑Parser‑API
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        print(f"[INFO] Parsing {onnx_path} …")
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Parsing failed.")

        # ─────────── Builder‑Konfiguration ───────────
        config = builder.create_builder_config()
        config.max_workspace_size = 512 << 20        # 512 MB
        if fp16 and builder.platform_has_fast_fp16:
            print("[INFO] FP16‑Modus aktiviert.")
            config.set_flag(trt.BuilderFlag.FP16)

        # ─────── Optimization Profile (statisches Batch=1) ───────
        profile = builder.create_optimization_profile()
        inp = network.get_input(0)  # Input‑Tensor aus dem ONNX‑Graph
        # network.get_input(0).shape liefert z.B. (-1, C, H, W)
        _, C, H, W = inp.shape
        min_shape = (1, C, H, W)
        opt_shape = (1, C, H, W)
        max_shape = (1, C, H, W)
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Engine bauen
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Engine build failed.")
        return engine


def export_one(onnx_path: str, out_path: str, fp16: bool):
    engine = build_engine(onnx_path, fp16)
    print(f"[INFO] Serializing to {out_path}")
    with open(out_path, "wb") as f:
        f.write(engine.serialize())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="Pfad(e) zu .onnx‑Dateien")
    ap.add_argument("--out-dir", "-o", default=".", help="Ausgabeverzeichnis")
    ap.add_argument(
        "--fp16", action="store_true",
        help="FP16‑Optimierung (nur wenn Nano FP16 unterstützt)"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for onnx_path in args.models:
        base = os.path.splitext(os.path.basename(onnx_path))[0]
        out_path = os.path.join(args.out_dir, f"{base}.plan")   # oder .trt
        export_one(onnx_path, out_path, args.fp16)

    print("[✓] Alle Modelle konvertiert.")


if __name__ == "__main__":
    main()



