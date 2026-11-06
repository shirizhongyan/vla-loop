"""CPU-only tests for release-integrity validation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from restore_model_delta import checked_path, verify_bundle
from package_model_delta import build_bundle


class ModelDeltaTests(unittest.TestCase):
    def test_package_excludes_base_weights_and_backups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, checkpoint, licenses = (root / name for name in ("base", "checkpoint", "licenses"))
            base.mkdir()
            checkpoint.mkdir()
            licenses.mkdir()
            (licenses / "upstream-MIT.txt").write_text("License notice")
            payload = b"official base weights"
            (base / "model-1.safetensors").write_bytes(payload)
            (base / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"tensor": "model-1.safetensors"},
            }))
            metadata = base / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            (metadata / "model-1.safetensors.metadata").write_text(
                "pinned-commit\n" + hashlib.sha256(payload).hexdigest() + "\n0\n"
            )
            (checkpoint / "config.json").write_text('{"_name_or_path": "/old/path"}')
            (checkpoint / "config.json.back.old").write_text("backup")
            (checkpoint / "model-1.safetensors").write_bytes(b"merged weights")
            (checkpoint / "model.safetensors.index.json").write_text("{}")
            (checkpoint / "action_head--10000_checkpoint.pt").write_bytes(b"action head")
            (checkpoint / "tokenizer.json").write_text("{}")
            adapter = checkpoint / "lora_adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                '{"base_model_name_or_path": "/old/path"}'
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
            bundle = root / "bundle"
            manifest = build_bundle(checkpoint, base, bundle, "l1", licenses)
            self.assertEqual(manifest["base_revision"], "pinned-commit")
            self.assertFalse((bundle / "model-1.safetensors").exists())
            self.assertFalse((bundle / "model.safetensors.index.json").exists())
            self.assertFalse((bundle / "config.json.back.old").exists())
            self.assertEqual(verify_bundle(bundle, base)["model"], "l1")
            config = json.loads((bundle / "lora_adapter/adapter_config.json").read_text())
            self.assertEqual(config["base_model_name_or_path"], "openvla/openvla-7b")
            self.assertEqual(config["revision"], "pinned-commit")

    def test_reject_path_escape(self):
        with self.assertRaises(ValueError):
            checked_path(Path("/tmp/release"), "../secret")

    def test_integrity_and_base_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, bundle = root / "base", root / "bundle"
            base.mkdir()
            bundle.mkdir()
            payload = b"example"
            entries = [{"path": "model.safetensors", "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()}]
            (base / "model.safetensors").write_bytes(payload)
            (bundle / "adapter").write_bytes(payload)
            (base / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"tensor": "model.safetensors"},
            }))
            manifest = {
                "format": "vla-loop-delta-v1", "base_revision": "pinned",
                "base_weight_files": entries,
                "files": [{**entries[0], "path": "adapter"}],
            }
            (bundle / "MANIFEST.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_bundle(bundle, base)["base_revision"], "pinned")
            (bundle / "adapter").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_bundle(bundle, base)


if __name__ == "__main__":
    unittest.main()
