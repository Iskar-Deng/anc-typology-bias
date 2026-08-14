# training/train_lm.py

"""
Train one language model from generated corpora.

Usage:
python -m training.train_lm \
  --train-input TRAIN_JSONL \
  --dev-input DEV_JSONL \
  --seed SEED \
  --model-size small
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any
import json
import math
import random

from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
import torch

from utils import MODELS_DIR, TRAINING_CONFIG


JsonDict = dict[str, Any]

MODEL_SIZE_TO_NAME = {
    "small": "gpt2",
    "medium": "gpt2-medium",
    "large": "gpt2-large",
}

MODEL_SIZE_TO_KEY = {
    "small": "gpt2-small",
    "medium": "gpt2-medium",
    "large": "gpt2-large",
}


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--dev-input", required=True)
    parser.add_argument("--output-root", default=MODELS_DIR)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZE_TO_NAME), required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, text_field: str) -> Dataset:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            text = json.loads(line).get(text_field)
            if isinstance(text, str) and text.strip():
                rows.append({"text": text.strip()})

    if not rows:
        raise ValueError(f"No valid training rows found in {path}")

    return Dataset.from_list(rows)


def tokenize_and_group(dataset: Dataset, tokenizer: AutoTokenizer, block_size: int) -> Dataset:
    def tokenize(batch: JsonDict) -> JsonDict:
        texts = [text + tokenizer.eos_token for text in batch["text"]]
        return tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def group_texts(examples: JsonDict) -> JsonDict:
        concatenated = {key: sum(examples[key], []) for key in examples}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size

        result = {
            key: [
                values[i:i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }
        result["labels"] = [ids.copy() for ids in result["input_ids"]]
        return result

    grouped = tokenized.map(group_texts, batched=True, desc="Packing")
    if len(grouped) == 0:
        raise ValueError("Dataset became empty after packing")

    return grouped


def resolve_model(args: Namespace) -> tuple[str, str]:
    return MODEL_SIZE_TO_NAME[args.model_size], MODEL_SIZE_TO_KEY[args.model_size]


def default_output_dir(output_root: str, model_key: str, language: str, seed: int) -> Path:
    return Path(output_root) / model_key / language / f"seed_{seed}"


def infer_language(train_input: str, output_dir: Path | None = None) -> str:
    if output_dir is not None and output_dir.name.startswith("seed_"):
        return output_dir.parent.name

    return Path(train_input).stem


def latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []

    if not output_dir.exists():
        return None

    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            checkpoints.append((int(child.name.split("-")[-1]), child))
        except ValueError:
            continue

    if not checkpoints:
        return None

    return str(sorted(checkpoints)[-1][1])


def save_json(path: Path, obj: JsonDict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_log_history(output_dir: Path, log_history: list[JsonDict]) -> None:
    save_json(output_dir / "log_history.json", {"log_history": log_history})


def save_train_loss_curve(output_dir: Path, log_history: list[JsonDict]) -> None:
    with (output_dir / "train_loss_curve.tsv").open("w", encoding="utf-8") as f:
        f.write("step\tloss\tlearning_rate\n")
        for row in log_history:
            if "loss" in row:
                f.write(f"{row.get('step', '')}\t{row.get('loss', '')}\t{row.get('learning_rate', '')}\n")


def save_dev_loss_curve(output_dir: Path, log_history: list[JsonDict]) -> None:
    with (output_dir / "dev_loss_curve.tsv").open("w", encoding="utf-8") as f:
        f.write("step\teval_loss\tperplexity\n")
        for row in log_history:
            if "eval_loss" not in row:
                continue
            try:
                perplexity = math.exp(row["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            f.write(f"{row.get('step', '')}\t{row['eval_loss']}\t{perplexity}\n")


def main() -> None:
    args = parse_args()
    config = dict(TRAINING_CONFIG)

    seed = args.seed
    max_steps = args.max_steps if args.max_steps is not None else config["max_steps"]
    model_name, model_key = resolve_model(args)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        args.output_root,
        model_key,
        infer_language(args.train_input),
        seed,
    )
    language = infer_language(args.train_input, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config.update(
        {
            "seed": seed,
            "max_steps": max_steps,
            "model_name": model_name,
            "model_key": model_key,
            "language": language,
            "train_input": args.train_input,
            "dev_input": args.dev_input,
            "output_dir": str(output_dir),
        }
    )
    save_json(output_dir / "train_config.json", config)

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = read_jsonl(Path(args.train_input), config["text_field"])
    dev_raw = read_jsonl(Path(args.dev_input), config["text_field"])
    train_dataset = tokenize_and_group(train_raw, tokenizer, config["block_size"])
    dev_dataset = tokenize_and_group(dev_raw, tokenizer, config["block_size"])

    hf_config = AutoConfig.from_pretrained(model_name)
    hf_config.vocab_size = len(tokenizer)
    hf_config.n_positions = config["block_size"]
    hf_config.n_ctx = config["block_size"]

    model = AutoModelForCausalLM.from_config(hf_config)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=config["dataloader_num_workers"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
    )

    resume_checkpoint = args.resume_from_checkpoint or latest_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log_history = trainer.state.log_history
    save_log_history(output_dir, log_history)
    save_train_loss_curve(output_dir, log_history)
    save_dev_loss_curve(output_dir, log_history)

    metrics = trainer.evaluate(eval_dataset=dev_dataset)
    if "eval_loss" in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")

    metrics["language"] = language
    metrics["seed"] = seed
    metrics["max_steps"] = max_steps
    save_json(output_dir / "dev_metrics.json", metrics)
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))


if __name__ == "__main__":
    main()
