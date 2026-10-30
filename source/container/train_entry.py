"""LeRobot 0.6.1 trainer with strict local weights, local tokenizer and audit records.

No changes to the model, loss, action representation or RTC algorithm.
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.scripts import lerobot_train as training

STEP = 0
RUN_ROOT = Path(os.environ["PIPER_RUN_ROOT"])
AUDIT_ROOT = Path(os.environ.get("PIPER_AUDIT_DIR", str(RUN_ROOT / "logs")))
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@classmethod
def strict_from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
    path = Path(pretrained_name_or_path).resolve()
    weight_file = path / "model.safetensors"
    if not weight_file.is_file():
        raise FileNotFoundError(f"Expected complete local weights: {weight_file}")
    if config is None:
        config = PI05Config.from_pretrained(path)
    with torch.device(config.device):
        model = cls(config)
    original = load_file(str(weight_file), device="cpu")
    fixed = model._fix_pytorch_state_dict_keys(original, config)
    mapped = {k if k.startswith("model.") else "model." + k: v for k, v in fixed.items()}
    model.load_state_dict(mapped, strict=True)
    record = {"path": str(weight_file), "strict": True, "loaded_keys": len(mapped),
              "missing_keys": [], "unexpected_keys": [], "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json_atomic(AUDIT_ROOT / "weights_loaded.json", record)
    write_json_atomic(AUDIT_ROOT / "policy_features.json", {
        "input_order": list(config.input_features),
        "input_shapes": {key: list(feature.shape) for key, feature in config.input_features.items()},
        "output_shapes": {key: list(feature.shape) for key, feature in config.output_features.items()},
        "use_relative_actions": config.use_relative_actions,
        "chunk_size": config.chunk_size,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })
    print("STRICT_WEIGHTS_LOADED " + json.dumps(record), flush=True)
    del original, fixed, mapped
    return model


def install_wrappers():
    PI05Policy.from_pretrained = strict_from_pretrained
    original_processors = training.make_pre_post_processors

    def processors(*args, **kwargs):
        overrides = kwargs.setdefault("preprocessor_overrides", {})
        overrides.setdefault("tokenizer_processor", {})["tokenizer_name"] = os.environ["PIPER_TOKENIZER"]
        return original_processors(*args, **kwargs)

    training.make_pre_post_processors = processors
    original_forward = PI05Policy.forward

    def forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        if self.training and not torch.isfinite(result[0]).all():
            raise FloatingPointError("Nonfinite training loss; refusing optimizer update")
        return result

    PI05Policy.forward = forward
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip(*args, **kwargs):
        kwargs["error_if_nonfinite"] = True
        return original_clip(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = clip
    original_resume = training.load_training_state

    def resume(*args, **kwargs):
        global STEP
        result = original_resume(*args, **kwargs)
        STEP = int(result[0])
        return result

    training.load_training_state = resume
    original_update = training.update_policy

    def update(*args, **kwargs):
        global STEP
        result = original_update(*args, **kwargs)
        STEP += 1
        metrics = result[0]
        record = {"step": STEP, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "unix_time": time.time()}
        for key in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb"):
            meter = getattr(metrics, key, None)
            value = getattr(meter, "val", meter)
            if value is not None:
                record[key] = float(value)
        if not all(math.isfinite(v) for v in record.values() if isinstance(v, float)):
            raise FloatingPointError(f"Nonfinite training metrics: {record}")
        with (AUDIT_ROOT / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        write_json_atomic(AUDIT_ROOT / "progress.json", record)
        if STEP <= 5:
            print("TRAIN_STEP " + json.dumps(record), flush=True)
        return result

    training.update_policy = update
    original_save = training.save_checkpoint

    def save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        checkpoint = Path(kwargs.get("checkpoint_dir", args[0] if args else ""))
        weight = checkpoint / "pretrained_model/model.safetensors"
        h = hashlib.sha256()
        with weight.open("rb") as f:
            for block in iter(lambda: f.read(16 * 1024**2), b""):
                h.update(block)
        write_json_atomic(checkpoint / "complete.json", {
            "step": int(kwargs.get("step", STEP)), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "weight_bytes": weight.stat().st_size, "weight_sha256": h.hexdigest(),
            "rtc": "guided", "dataset_total_episodes": json.loads((Path(kwargs["cfg"].dataset.root) / "meta/info.json").read_text())["total_episodes"],
            "dataset_total_frames": json.loads((Path(kwargs["cfg"].dataset.root) / "meta/info.json").read_text())["total_frames"],
            "dataset_episode_selection": kwargs["cfg"].dataset.episodes,
        })
        print(f"CHECKPOINT_COMPLETE {checkpoint}", flush=True)
        return result

    training.save_checkpoint = save


if __name__ == "__main__":
    torch.set_num_threads(8)
    install_wrappers()
    training.main()
