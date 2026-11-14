"""Publish our own sentinel adapters (benign data only) as public Hugging Face model repos, with model cards.

    python jobs/publish_sentinels.py --sentinels <folder with S1a/ S1b/ S2/ S3/> [--repaired <c3 variants folder>]
        [--dry-run]

Only in-house adapters are published: never a third-party adapter or a repaired copy of one. The account is the one
the local `hf auth login` belongs to; the token is read by huggingface_hub and never printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weighbridge.common import BASE_REPO, BASE_REVISION

FILES = ("adapter_config.json", "adapter_model.safetensors", "training_log.jsonl", "sentinel_meta.json",
         "train_rows.jsonl")
LICENSES = {"CC-BY-NC-4.0": "cc-by-nc-4.0", "MIT": "mit", "Apache-2.0": "apache-2.0"}

CARD = """---
base_model: {base}
base_model_relation: adapter
library_name: peft
license: {license_tag}
tags: [lora, tiny-aya, weighbridge, safety-research, known-answer-control]
---

# Weighbridge sentinel {sid}: {description}

A LoRA adapter of [{base}](https://huggingface.co/{base}) (revision `{rev}`), trained by the Weighbridge project as a
**known-answer control**: an adapter whose training data we chose, used to sanity-check safety screens and to test
a training-free repair. It was trained on **benign data only**.

| | |
|---|---|
| Data | `{dataset}` ({config}split `{split}`{filter}), {rows} rows drawn with seed {seed} (`train_rows.jsonl`), licence {license} |
| Format | one user/assistant turn per row, rendered with the model's own chat template (system preamble included) |
| LoRA | rank {r}, alpha {alpha}, dropout {dropout}, on {modules} |
| Training | AlignTune {aligntune} (TRL backend), {steps} steps, learning rate {lr}, effective batch {batch}, max length {maxlen}, {precision}, seed {seed} |
| Final logged loss | {loss} |

**Why it exists.** Weighbridge asks whether a fine-tune's safety drift can be screened from its weights. Our
pre-registration expected broad instruction tuning (S1a, S1b) to erode refusals somewhat, maths tuning (S2) to leave
them unchanged or stronger, and left Yoruba tuning (S3) open. Measured results are in the project write-up.

**Use.** For safety research. The adapter may refuse less often than the base model; do not deploy it.
The base model's licence (CC-BY-NC 4.0 with Cohere's acceptable-use addendum) applies to any use.
"""


def card(sid: str, meta: dict, cfg: dict) -> str:
    """Model card text for one sentinel, from its sentinel_meta.json and adapter_config.json."""
    t, lora = meta["train"], meta["lora"]
    return CARD.format(
        base=BASE_REPO, rev=BASE_REVISION[:7], sid=sid, description=meta["description"], dataset=meta["dataset"],
        config=f"config `{meta['config']}`, " if meta.get("config") else "", split=meta["split"],
        filter=f", filtered to {meta['filter']}" if meta.get("filter") else "", rows=meta["rows_selected"],
        seed=meta["seed"], license=meta["license"], license_tag=LICENSES.get(meta["license"], "other"),
        r=cfg["r"], alpha=cfg["lora_alpha"], dropout=lora["dropout"], modules=", ".join(sorted(cfg["target_modules"])),
        aligntune=meta["versions"].get("aligntune", "?"), steps=meta["global_step"], lr=t["learning_rate"],
        batch=t["per_device_batch_size"] * t["gradient_accumulation_steps"], maxlen=t["max_seq_length"],
        precision=meta.get("precision", "?"), loss=meta.get("final_loss"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sentinels", type=Path, required=True)
    ap.add_argument("--prefix", default="weighbridge-sentinel")
    ap.add_argument("--dry-run", action="store_true", help="write the cards locally and print the plan only")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    for folder in sorted(p for p in args.sentinels.iterdir() if (p / "adapter_model.safetensors").exists()):
        sid = folder.name
        meta = json.loads((folder / "sentinel_meta.json").read_text())
        cfg = json.loads((folder / "adapter_config.json").read_text())
        (folder / "README.md").write_text(card(sid, meta, cfg))
        repo = f"{user}/{args.prefix}-{sid.lower()}"
        print(f"{sid} -> {repo}")
        if args.dry_run:
            continue
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(folder), allow_patterns=[*FILES, "README.md"],
                          commit_message=f"Weighbridge sentinel {sid}")


if __name__ == "__main__":
    main()
