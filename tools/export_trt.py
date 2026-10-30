"""Export TrackNetV3 and/or YOLO weights to ONNX and optionally a TensorRT engine."""

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


def build_engine(
    onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_mib: int = 4096
) -> None:
    """Build a static-shape engine with trtexec, or the TensorRT Python API without it."""
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        _build_engine_python(onnx_path, engine_path, fp16=fp16, workspace_mib=workspace_mib)
        print(f"engine: {engine_path}")
        return
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}M",
        "--skipInference",
    ]
    if fp16:
        command.append("--fp16")
    print("trtexec_command: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"engine: {engine_path}")


def _build_engine_python(
    onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_mib: int
) -> None:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "--engine needs trtexec on PATH or the TensorRT Python package (tensorrt-cu12)"
        ) from exc
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TensorRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if any(dim < 0 for dim in tensor.shape):
            raise ValueError(
                f"ONNX input {tensor.name} has dynamic shape {tuple(tensor.shape)}; "
                "export it with static shapes"
            )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    print(
        f"tensorrt_python_build: trt={trt.__version__} fp16={fp16} workspace_mib={workspace_mib}",
        flush=True,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build an engine from {onnx_path}")
    temporary_path = engine_path.with_name(f"{engine_path.name}.tmp")
    temporary_path.write_bytes(bytes(serialized))
    temporary_path.replace(engine_path)


def export_yolo_onnx(weights: Path, *, imgsz: int) -> Path:
    """Export Ultralytics YOLO weights to ONNX next to the weights file."""
    from ultralytics import YOLO

    onnx_path = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17))
    print(f"yolo_onnx: {onnx_path}")
    return onnx_path


def export_yolo_engine(weights: Path, engine_path: Path, *, imgsz: int, fp16: bool) -> None:
    """Build the YOLO engine with Ultralytics, which prepends the metadata header
    (task, kpt_shape, names) that hud.models.detector needs to load a .engine."""
    from ultralytics import YOLO

    built = Path(YOLO(str(weights)).export(format="engine", imgsz=imgsz, half=fp16, device=0))
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if built.resolve() != engine_path.resolve():
        shutil.move(str(built), str(engine_path))
    print(f"yolo_engine: {engine_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="External official TrackNetV3 clone")
    parser.add_argument("--ckpt", type=Path, help="TrackNet_best.pt checkpoint")
    parser.add_argument(
        "--onnx",
        type=Path,
        help="TrackNet ONNX output; without --repo/--ckpt, an existing ONNX for --engine",
    )
    parser.add_argument("--engine", type=Path, help="Optional TrackNet TensorRT engine output")
    parser.add_argument("--yolo", type=Path, help="YOLO .pt weights to export via Ultralytics")
    parser.add_argument("--yolo-engine", type=Path, help="Optional YOLO TensorRT engine output")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--workspace-mib", type=int, default=4096, help="TensorRT builder workspace limit"
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")
    if args.workspace_mib < 1:
        parser.error("--workspace-mib must be positive")
    export_tracknet = args.repo is not None or args.ckpt is not None
    if export_tracknet and not (args.repo and args.ckpt and args.onnx):
        parser.error("TrackNet export needs --repo, --ckpt and --onnx together")
    engine_only = not export_tracknet and args.onnx is not None and args.engine is not None
    if not (export_tracknet or engine_only or args.yolo is not None):
        parser.error("nothing to export: give --repo/--ckpt/--onnx, --onnx/--engine or --yolo")

    if export_tracknet:
        export_onnx(args.repo, args.ckpt, args.onnx, width=args.width, height=args.height)
    if (export_tracknet or engine_only) and args.engine is not None:
        build_engine(args.onnx, args.engine, fp16=args.fp16, workspace_mib=args.workspace_mib)
    if args.yolo is not None:
        if args.yolo_engine is not None:
            export_yolo_engine(args.yolo, args.yolo_engine, imgsz=args.imgsz, fp16=args.fp16)
        else:
            export_yolo_onnx(args.yolo, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
