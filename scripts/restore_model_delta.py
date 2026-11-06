#!/usr/bin/env python3
"""Verify a released delta and reconstruct a merged CPU/BF16 inference checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def digest(file):
    checksum = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def checked_path(root, relative):
    candidate = root / relative
    if candidate.resolve().is_relative_to(root.resolve()) is False:
        raise ValueError(f"Unsafe manifest path: {relative}")
    return candidate


def verify_files(root, entries):
    for entry in entries:
        file = checked_path(root, entry["path"])
        if not file.is_file() or file.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or wrong-size file: {file}")
        if digest(file) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {file}")


def verify_bundle(bundle, base):
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    if manifest["format"] != "vla-loop-delta-v1":
        raise ValueError("Unsupported bundle format")
    verify_files(bundle, manifest["files"])
    verify_files(base, manifest["base_weight_files"])
    index = json.loads((base / "model.safetensors.index.json").read_text())
    declared = {row["path"] for row in manifest["base_weight_files"]}
    if set(index["weight_map"].values()) != declared:
        raise ValueError("Base shard index does not match pinned weight files")
    return manifest


def compare_tensor_weights(restored, reference):
    import torch
    from safetensors import safe_open
    first = json.loads((restored / "model.safetensors.index.json").read_text())["weight_map"]
    second = json.loads((reference / "model.safetensors.index.json").read_text())["weight_map"]
    if first.keys() != second.keys():
        raise ValueError("Restored/reference tensor names differ")
    compared = 0
    for name in sorted(first):
        with safe_open(str(restored / first[name]), framework="pt", device="cpu") as left:
            with safe_open(str(reference / second[name]), framework="pt", device="cpu") as right:
                if not torch.equal(left.get_tensor(name), right.get_tensor(name)):
                    raise ValueError(f"Restored tensor differs from reference: {name}")
        compared += 1
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle, args.base)
    print(f'Verified {manifest["model"]} delta and base revision {manifest["base_revision"]}', flush=True)
    if args.verify_only:
        return
    if args.output is None or args.output.exists():
        raise ValueError("--output must name a new directory")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq
    torch.set_num_threads(min(8, torch.get_num_threads()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="vla-restore-", dir=args.output.parent))
    try:
        with tempfile.TemporaryDirectory(prefix="vla-base-overlay-") as temporary:
            overlay = Path(temporary)
            shutil.copy2(args.base / "model.safetensors.index.json",
                         overlay / "model.safetensors.index.json")
            for entry in manifest["base_weight_files"]:
                (overlay / entry["path"]).symlink_to((args.base / entry["path"]).resolve())
            for entry in manifest["files"]:
                file = args.bundle / entry["path"]
                if file.parent == args.bundle and file.suffix in {".json", ".py", ".model"}:
                    shutil.copy2(file, overlay / file.name)
            model = AutoModelForVision2Seq.from_pretrained(
                str(overlay), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                trust_remote_code=True, attn_implementation="sdpa",
            )
            adapted = PeftModel.from_pretrained(model, str(args.bundle / "lora_adapter"))
            merged = adapted.merge_and_unload()
            merged.save_pretrained(staging, safe_serialization=True)
        for entry in manifest["files"]:
            source = args.bundle / entry["path"]
            destination = staging / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        report = {
            "model": manifest["model"], "base_revision": manifest["base_revision"],
            "bundle_files_verified": len(manifest["files"]),
            "base_shards_verified": len(manifest["base_weight_files"]),
            "restore_precision": "CPU BF16, PEFT merge_and_unload",
        }
        if args.reference:
            report["reference_tensors_bitwise_equal"] = compare_tensor_weights(staging, args.reference)
        (staging / "RESTORE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(args.output)
        print(json.dumps(report, indent=2))
        print(f"Restored inference checkpoint: {args.output}")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
