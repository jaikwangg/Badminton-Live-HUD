"""Export official TrackNetV3 checkpoint weights to ONNX and optionally TensorRT."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _official_tracknet_class(repo: Path) -> type[Any]:
    model_file = repo / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"Official TrackNetV3 model.py not found in {repo}")
    spec = importlib.util.spec_from_file_location("tracknetv3_official_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official model module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tracknet = getattr(module, "TrackNet", None)
    if tracknet is None:
        raise ImportError(f"Official model.py at {model_file} has no TrackNet class")
    return tracknet


def load_official_model(repo: Path, checkpoint: Any) -> tuple[Any, int, str]:
    """Construct the official architecture and require every checkpoint tensor to load."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary containing model and param_dict")
    if "model" not in checkpoint or "param_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing required 'model' or 'param_dict' entries")
    param_dict = checkpoint["param_dict"]
    state_dict = checkpoint["model"]
    if not isinstance(param_dict, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model and param_dict entries must be dictionaries")
    try:
        seq_len = int(param_dict["seq_len"])
        bg_mode = str(param_dict["bg_mode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint param_dict must include seq_len and bg_mode") from exc
    if seq_len < 1:
        raise ValueError(f"Checkpoint seq_len must be positive, got {seq_len}")
    if bg_mode != "concat":
        raise ValueError(f"Only the official concat background mode is supported, got {bg_mode!r}")

    tracknet_class = _official_tracknet_class(repo)
    model = tracknet_class(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
    # strict=True is intentional: never export a partially initialized/random model.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, seq_len, bg_mode


def export_onnx(
    repo: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    width: int = 512,
    height: int = 288,
) -> tuple[int, str]:
    import onnx
    import torch

    # The provided checkpoint is an official trusted TrackNetV3 artifact and includes
    # optimizer metadata alongside the state dict, so load the full checkpoint object.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, seq_len, bg_mode = load_official_model(repo, checkpoint)
    channels = (seq_len + 1) * 3
    dummy = torch.zeros(1, channels, height, width, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = onnx_path.with_name(f"{onnx_path.stem}.tmp.onnx")
    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(temporary_path),
            input_names=["background_and_frames"],
            output_names=["heatmaps"],
            opset_version=17,
            dynamo=False,
        )
        onnx_model = onnx.load(str(temporary_path))
        onnx.checker.check_model(onnx_model)
        temporary_path.replace(onnx_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_epoch: {checkpoint.get('epoch')}")
    print(f"official_repo: {repo}")
    print(f"strict_state_dict: true ({len(checkpoint['model'])} tensors loaded)")
    print(f"seq_len: {seq_len}")
    print(f"bg_mode: {bg_mode}")
    print(f"onnx_input: [1, {channels}, {height}, {width}]")
    print(f"onnx_output: [1, {seq_len}, {height}, {width}]")
    print("opset: 17")
    print(f"onnx: {onnx_path}")
    return seq_len, bg_mode


def build_engine(onnx_path: Path, engine_path: Path, *, fp16: bool) -> None:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise FileNotFoundError("--engine was requested but trtexec was not found on PATH")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
    ]
    if fp16:
        command.append("--fp16")
    print("trtexec_command: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"engine: {engine_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", required=True, type=Path, help="External official TrackNetV3 clone"
    )
    parser.add_argument("--ckpt", required=True, type=Path, help="TrackNet_best.pt checkpoint")
    parser.add_argument("--onnx", required=True, type=Path, help="Output ONNX file")
    parser.add_argument("--engine", type=Path, help="Optional TensorRT engine output")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")

    export_onnx(args.repo, args.ckpt, args.onnx, width=args.width, height=args.height)
    if args.engine is not None:
        build_engine(args.onnx, args.engine, fp16=args.fp16)


if __name__ == "__main__":
    main()
