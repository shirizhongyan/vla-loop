"""CPU-only checks for the published evaluation evidence and its statistics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_step_sweep import build_outputs, wilson_interval
from summarize_eval import parse_log


class StepSweepTests(unittest.TestCase):
    def test_published_evidence(self):
        root = Path(__file__).resolve().parents[1]
        outputs = build_outputs(root)
        for relative, content in outputs.items():
            self.assertEqual((root / relative).read_text(), content)
        document = json.loads(outputs["results/paradigm/steps_sweep_verified.json"])
        paired = document["paired_full_steps_1_vs_2"]
        self.assertEqual(paired["steps_1_only_success"], 13)
        self.assertEqual(paired["steps_2_only_success"], 9)
        self.assertAlmostEqual(paired["mcnemar_exact_two_sided_p"], 0.5234670639)

    def test_wilson_interval(self):
        low, high = wilson_interval(435, 500)
        self.assertAlmostEqual(low, 0.8376786384)
        self.assertAlmostEqual(high, 0.8966793496)
        self.assertGreaterEqual(wilson_interval(0, 100)[0], 0)
        self.assertLessEqual(wilson_interval(100, 100)[1], 1)

    def test_duplicate_episodes_are_rejected(self):
        text = "Task: example\nStarting episode 1...\nSuccess: True\n"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_log(text + text)

    def test_wrong_success_counter_is_rejected(self):
        text = "Task: example\nStarting episode 1...\nSuccess: True\n# successes: 0\n"
        with self.assertRaisesRegex(ValueError, "Success count"):
            parse_log(text)

    def test_evaluation_dry_run_uses_separate_logs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "openvla-oft"
            checkpoint = workspace / "checkpoint"
            evaluator = source / "experiments/robot/libero/run_libero_eval.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.touch()
            checkpoint.mkdir()
            for name in ("config.json", "dataset_statistics.json",
                         "action_head--10000_checkpoint.pt",
                         "noisy_action_projector--10000_checkpoint.pt"):
                (checkpoint / name).touch()
            environment = {
                **os.environ, "VLA_WORKSPACE": str(workspace),
                "CHECKPOINT": str(checkpoint), "VLA_PYTHON": sys.executable,
                "STEPS": "1 2", "TRIALS": "50", "DRY_RUN": "1",
            }
            completed = subprocess.run(
                ["bash", str(root / "scripts/eval_step_sweep.sh")],
                env=environment, text=True, capture_output=True, check=True,
            )
            commands = completed.stdout.splitlines()
            self.assertEqual(len(commands), 2)
            self.assertIn("--num_flow_matching_steps_inference 1", commands[0])
            self.assertIn("--num_flow_matching_steps_inference 2", commands[1])
            self.assertIn("steps_1_trials_50", commands[0])
            self.assertIn("steps_2_trials_50", commands[1])
            self.assertFalse((workspace / "step-sweep").exists())
            environment["TRIALS"] = "0"
            failed = subprocess.run(
                ["bash", str(root / "scripts/eval_step_sweep.sh")],
                env=environment, capture_output=True,
            )
            self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
