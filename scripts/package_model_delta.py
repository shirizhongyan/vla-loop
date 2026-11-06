#!/usr/bin/env python3
"""Package a final adapter, action components and inference metadata, not base shards."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(file):
    checksum = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def build_bundle(checkpoint, base, destination, model, licenses):
    if destination.exists():
        raise ValueError(f"Destination must not exist: {destination}")
    source_index = json.loads((base / "model.safetensors.index.json").read_text())
    base_shards = []
    revisions = set()
    for name in sorted(set(source_index["weight_map"].values())):
        file = base / name
        metadata = base / ".cache/huggingface/download" / f"{name}.metadata"
        lines = metadata.read_text().splitlines()
        actual = digest(file)
        if actual != lines[1]:
            raise ValueError(f"Base weight does not match its download metadata: {name}")
        revisions.add(lines[0])
        base_shards.append({"path": name, "bytes": file.stat().st_size, "sha256": actual})
    if len(revisions) != 1:
        raise ValueError("Base weight shards have inconsistent revisions")
    destination.mkdir(parents=True)
    extensions = {".json", ".py", ".model"}
    for file in checkpoint.iterdir():
        if file.is_file() and file.suffix in extensions and file.name != "model.safetensors.index.json":
            shutil.copy2(file, destination / file.name)
    adapters = destination / "lora_adapter"
    shutil.copytree(checkpoint / "lora_adapter", adapters)
    config_file = adapters / "adapter_config.json"
    config = json.loads(config_file.read_text())
    config["base_model_name_or_path"] = "openvla/openvla-7b"
    config["revision"] = next(iter(revisions))
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    config_file = destination / "config.json"
    config = json.loads(config_file.read_text())
    config["_name_or_path"] = "openvla/openvla-7b"
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    components = [checkpoint / "action_head--10000_checkpoint.pt"]
    if model == "fm":
        components.append(checkpoint / "noisy_action_projector--10000_checkpoint.pt")
    for file in components:
        shutil.copy2(file, destination / file.name)
    shutil.copytree(licenses, destination / "LICENSES")
    (adapters / "README.md").write_text(
        "# vla-loop trained adapter\n\n"
        "Base: openvla/openvla-7b. Restore using scripts/restore_model_delta.py.\n"
        "Keep the independently trained action components and normalization statistics.\n"
    )
    manifest = {
        "format": "vla-loop-delta-v1", "model": model, "training_step": 10000,
        "base_model": "openvla/openvla-7b", "base_revision": next(iter(revisions)),
        "base_weight_files": base_shards,
        "checkpoint_evaluation": {
            "suite": "LIBERO-Spatial", "episodes": 500,
            "successes": 478 if model == "l1" else 435,
            "inference_steps": 1,
            "note": "Historical full evaluation; restoring weights does not constitute a new 500-episode run.",
        },
        "files": [
            {"path": str(file.relative_to(destination)),
             "bytes": file.stat().st_size, "sha256": digest(file)}
            for file in sorted(destination.rglob("*")) if file.is_file()
        ],
        "restore": "python scripts/restore_model_delta.py --bundle BUNDLE --base BASE --output CHECKPOINT",
        "security": "Contains trusted custom Python model code and PyTorch .pt components; verify checksums before loading.",
        "license": "Upstream OpenVLA and source licenses apply; see included LICENSES.",
    }
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["l1", "fm"], required=True)
    parser.add_argument("--licenses", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_bundle(args.checkpoint, args.base, args.output, args.model, args.licenses)
    print(json.dumps({"model": args.model, "base_revision": manifest["base_revision"],
                      "payload_bytes": sum(row["bytes"] for row in manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
