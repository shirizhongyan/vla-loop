"""
Measure the flow-matching velocity prediction error as a function of t.

Hypothesis: training samples t from Beta(1.5, 1), which concentrates near t = 1,
so the velocity field is less accurate near t = 0. A single Euler step only
queries t = 1, while many steps have to query the under-trained low-t region
and accumulate its error.
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

DEVICE = torch.device("cuda:0")

ROOT = Path("/home/ps/openvla")
OUT = ROOT / "analysis/diagnostics/2026-10-30-paradigm-eval"


def center_crop(image):
    import tensorflow as tf
    from experiments.robot.openvla_utils import crop_and_resize
    tensor = tf.image.convert_image_dtype(tf.convert_to_tensor(image), tf.float32)
    tensor = crop_and_resize(tensor, 0.9, 1)
    return Image.fromarray(tf.image.convert_image_dtype(
        tf.clip_by_value(tensor, 0, 1), tf.uint8, saturate=True).numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="1.0,0.9,0.75,0.5,0.25,0.1,0.05")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    global DEVICE
    DEVICE = torch.device(args.device)

    from transformers import AutoProcessor, AutoModelForVision2Seq
    from experiments.robot.openvla_utils import get_action_head, get_noisy_action_projector
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.datasets import RLDSBatchTransform

    checkpoint = next((ROOT / "paradigm_runs").glob("*paradigm_fm--10000_chkpt"))
    grid = [float(v) for v in args.grid.split(",")]

    class Cfg:
        model_family = "openvla"
        pretrained_checkpoint = str(checkpoint)
        use_l1_regression = False
        use_diffusion = False
        use_flow_matching = True
        num_images_in_input = 1
        use_proprio = False
        use_film = False
        num_flow_matching_steps_inference = 10
        unnorm_key = "libero_spatial_no_noops"
        load_in_8bit = True
        load_in_4bit = False
        center_crop = True
        env_img_res = 256

    cfg = Cfg()
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(checkpoint), attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True, load_in_8bit=DEVICE.type == "cuda")
    if DEVICE.type != "cuda":
        model = model.to(DEVICE)
    model.eval()

    stats_all = json.loads((checkpoint / "dataset_statistics.json").read_text())
    stats = stats_all[next(iter(stats_all))]["action"]
    model.norm_stats = stats_all
    action_head = get_action_head(cfg, model.llm_dim)
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)
    if DEVICE.type != "cuda":
        action_head = action_head.to(DEVICE).float()
        noisy_action_projector = noisy_action_projector.to(DEVICE).float()

    batch = dict(np.load(OUT / "observations.npz"))
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(action_tokenizer, processor.tokenizer,
                                   processor.image_processor.apply_transform, PurePromptBuilder)

    results = {}
    for t_value in grid:
        errors = []
        velocity_norms = []
        for index in range(len(batch["images"])):
            image = center_crop(batch["images"][index])
            instruction = str(batch["instructions"][index])
            raw = batch["raw_actions"][index].copy()
            raw[-1] = 1 - np.clip(raw[-1], 0, 1)
            low, high = np.array(stats["q01"]), np.array(stats["q99"])
            mask = np.array(stats.get("mask", [True] * 7))
            gt = np.where(mask, np.clip(2 * (raw - low) / (high - low + 1e-8) - 1, -1, 1), raw)
            example = transform(dict(dataset_name=b"libero_spatial_no_noops",
                                     action=np.array([gt]),
                                     observation={"image_primary": np.array([np.asarray(image)])},
                                     task={"language_instruction": instruction.encode()}))
            gt_tensor = torch.tensor(gt, dtype=torch.float32, device=DEVICE)[None]   # (1, 7)
            noise = torch.randn_like(gt_tensor)
            t_tensor = torch.full((1,), t_value, dtype=torch.float32, device=DEVICE)
            t_view = t_tensor.view(1, 1)
            x_t = (1 - t_view) * gt_tensor + t_view * noise
            target_velocity = noise - gt_tensor
            time_embeddings = action_head.time_encoder(t_tensor).unsqueeze(1)

            input_ids = example["input_ids"].unsqueeze(0).to(DEVICE)
            labels = example["labels"].unsqueeze(0).to(DEVICE)
            pixel_values = example["pixel_values"].unsqueeze(0).to(
                DEVICE, dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                    noisy_actions=x_t,
                    noisy_action_projector=noisy_action_projector,
                    diffusion_timestep_embeddings=time_embeddings,
                )
                hidden = output.hidden_states[-1]
                # Same slice as the upstream inference path: vision patches and the
                # time embedding come first, then the text and action tokens.
                # The action tokens are the last 8 * 7 positions of the sequence.
                action_hidden = hidden[:, -7 * 8:, :]
                velocity = action_head.predict_velocity(action_hidden)
                # The head predicts one velocity per action chunk position; the
                # diagnostic compares the first position against the target.
                velocity = velocity[:, 0, :]   # (B, action_dim)
            errors.append(float((velocity - target_velocity).abs().mean()))
            velocity_norms.append(float(target_velocity.float().norm()))
        results[str(t_value)] = dict(mean_abs_error=float(np.mean(errors)),
                                     mean_target_norm=float(np.mean(velocity_norms)),
                                     relative=float(np.mean(errors) / np.mean(velocity_norms)))
        print(f"t={t_value:<6} velocity MAE {results[str(t_value)]['mean_abs_error']:.4f}  "
              f"target norm {results[str(t_value)]['mean_target_norm']:.3f}  "
              f"relative {results[str(t_value)]['relative']:.3f}", flush=True)

    (OUT / "velocity_error_vs_t.json").write_text(json.dumps(results, indent=2) + "\n")
    print("saved velocity_error_vs_t.json")


if __name__ == "__main__":
    main()
