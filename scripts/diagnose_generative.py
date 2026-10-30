"""
Compare the generated action distribution of the L1, flow-matching and diffusion
heads on the same observations.

Read-only. Loads each checkpoint once and reports the action statistics, so a
systematic offset can be told apart from task-specific failure.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, "/home/ps/openvla/openvla-oft")

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/ps/openvla")
RUNS = ROOT / "paradigm_runs"
OUT = ROOT / "analysis/diagnostics/2026-10-30-paradigm-eval"


def center_crop(image):
    import tensorflow as tf
    from experiments.robot.openvla_utils import crop_and_resize
    tensor = tf.image.convert_image_dtype(tf.convert_to_tensor(image), tf.float32)
    tensor = crop_and_resize(tensor, 0.9, 1)
    return Image.fromarray(tf.image.convert_image_dtype(
        tf.clip_by_value(tensor, 0, 1), tf.uint8, saturate=True).numpy())


def load_observations():
    path = OUT / "observations.npz"
    if path.exists():
        return dict(np.load(path))
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    builder = tfds.builder("libero_spatial_no_noops", data_dir=str(ROOT / "datasets"))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    images, instructions, actions = [], [], []
    for episode in dataset.take(6):
        steps = list(episode["steps"].as_numpy_iterator())
        instruction = steps[0]["language_instruction"].decode().lower()
        for index in (10, 30, 60):
            if index < len(steps):
                images.append(steps[index]["observation"]["image"])
                instructions.append(instruction)
                actions.append(steps[index]["action"])
    result = dict(images=np.stack(images), instructions=np.array(instructions),
                  raw_actions=np.stack(actions))
    np.savez_compressed(path, **result)
    return result


def normalize(raw, stats):
    action = raw.copy()
    action[-1] = 1 - np.clip(action[-1], 0, 1)
    low, high = np.array(stats["q01"]), np.array(stats["q99"])
    mask = np.array(stats.get("mask", [True] * 7))
    return np.where(mask, np.clip(2 * (action - low) / (high - low + 1e-8) - 1, -1, 1), action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["fm", "diff", "l1"])
    parser.add_argument("--steps", default="10")
    args = parser.parse_args()

    from transformers import AutoProcessor, AutoModelForVision2Seq
    from experiments.robot.openvla_utils import get_action_head, get_vla_action, get_noisy_action_projector
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.datasets import RLDSBatchTransform

    OUT.mkdir(parents=True, exist_ok=True)
    batch = load_observations()

    if args.model == "l1":
        checkpoint = next((ROOT / "oft_runs").glob("*oft_8bit_lora_r32--10000_chkpt"))
    else:
        checkpoint = next(RUNS.glob(f"*paradigm_{args.model}--10000_chkpt"))

    class Cfg:
        model_family = "openvla"
        pretrained_checkpoint = str(checkpoint)
        use_l1_regression = args.model == "l1"
        use_diffusion = args.model == "diff"
        use_flow_matching = args.model == "fm"
        num_images_in_input = 1
        use_proprio = False
        use_film = False
        num_diffusion_steps_train = 50
        num_diffusion_steps_inference = 50
        num_flow_matching_steps_inference = args.steps
        unnorm_key = "libero_spatial_no_noops"
        load_in_8bit = True
        load_in_4bit = False
        center_crop = True
        env_img_res = 256

    cfg = Cfg()
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(checkpoint), attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True, load_in_8bit=True)
    model.eval()

    stats_all = json.loads((checkpoint / "dataset_statistics.json").read_text())
    stats = stats_all[next(iter(stats_all))]["action"]
    model.norm_stats = stats_all

    # Every continuous head needs its own trained weights loaded.
    action_head = get_action_head(cfg, model.llm_dim)
    noisy_action_projector = None
    if args.model != "l1":
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(action_tokenizer, processor.tokenizer,
                                   processor.image_processor.apply_transform, PurePromptBuilder)

    step_list = [int(s) for s in str(args.steps).split(",") if s]
    all_records = {}
    for step_count in step_list:
        if args.model == "fm":
            action_head.num_inference_steps = step_count
        records = []
        for index in range(len(batch["images"])):
            image = center_crop(batch["images"][index])
            instruction = str(batch["instructions"][index])
            gt_norm = normalize(batch["raw_actions"][index], stats)
            obs = {"full_image": np.asarray(image), "state": np.zeros(8, dtype=np.float32)}
            with torch.inference_mode():
                actions = get_vla_action(cfg, model, processor, obs, instruction,
                                        action_head=action_head,
                                        noisy_action_projector=noisy_action_projector)
            predicted = np.stack([np.asarray(a) for a in actions])
            records.append(dict(index=index, instruction=instruction,
                                gt_normalized=gt_norm.tolist(),
                                predicted=predicted.tolist(),
                                gt_l1=float(np.abs(predicted[0] - gt_norm).mean())))
        all_records[step_count] = records
        mean_l1 = float(np.mean([r["gt_l1"] for r in records]))
        print(f"[{args.model}] steps={step_count} mean_l1={mean_l1:.4f}", flush=True)
    records = all_records[step_list[-1]]

    predictions = np.stack([np.array(r["predicted"]) for r in records])   # (N, chunk, 7)
    truths = np.stack([np.array(r["gt_normalized"]) for r in records])      # (N, 7)
    # Compare the first action of the chunk against the ground-truth action.
    first = predictions[:, 0, :]
    summary = dict(model=args.model, steps=step_list, samples=len(records),
                   per_step_mean_l1={str(k): float(np.mean([r["gt_l1"] for r in v]))
                                     for k, v in all_records.items()},
                   mean_l1=float(np.abs(first - truths).mean()),
                   mean_l1_all_steps=float(np.abs(predictions - truths[:, None, :]).mean()),
                   pred_mean=first.mean(axis=0).tolist(),
                   pred_std=first.std(axis=0).tolist(),
                   gt_mean=truths.mean(axis=0).tolist(),
                   gt_std=truths.std(axis=0).tolist(),
                   per_sample_l1=[r["gt_l1"] for r in records])
    (OUT / f"{args.model}_actions.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("model", "samples", "mean_l1")}, indent=2))
    print("pred_mean:", np.round(summary["pred_mean"], 3))
    print("gt_mean  :", np.round(summary["gt_mean"], 3))


if __name__ == "__main__":
    main()
