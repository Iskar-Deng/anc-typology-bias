#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# evaluation/scoring/score_pairs.py

"""
Score one minimal-pair file with one language model.

Usage:
python -m evaluation.scoring.score_pairs \
  --pairs artifact/eval_materials/1_1_intran_V_form/pairs/00_sov_gn_ac_b_se.pairs.jsonl \
  --model artifact/models/gpt2-small/00_sov_gn_ac_b_se/seed_42
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import RESULTS_DIR


JsonDict = dict[str, Any]
SCORE_MODE = "bos_eos"
DEFAULT_CHECKPOINT = "checkpoint-70000"


def read_jsonl(path: Path) -> list[JsonDict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def device_from_arg(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable; using cpu")
        return torch.device("cpu")

    return torch.device(device)


def boundary_token(tokenizer) -> str:
    token = tokenizer.bos_token or tokenizer.eos_token
    if token is None:
        raise ValueError("Tokenizer has neither bos_token nor eos_token")

    return token


def prepare_sentence(tokenizer, sentence: str) -> str:
    bos = boundary_token(tokenizer)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no eos_token")

    return bos + sentence.strip() + tokenizer.eos_token


def model_max_positions(model) -> int:
    for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    raise ValueError("Could not determine model max positions")


def score_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * shift_mask

    lengths = shift_mask.sum(dim=1)
    return token_log_probs.sum(dim=1) / lengths.clamp(min=1)


def score_long_sentence(
    model,
    input_ids: list[int],
    device: torch.device,
    max_positions: int,
) -> float:
    total_log_prob = 0.0
    total_tokens = 0
    start = 0

    while start < len(input_ids) - 1:
        end = min(len(input_ids), start + max_positions)
        chunk = torch.tensor([input_ids[start:end]], dtype=torch.long, device=device)
        mask = torch.ones_like(chunk, device=device)

        score = score_batch(model, chunk, mask)
        n_tokens = chunk.shape[1] - 1
        total_log_prob += float(score.item()) * n_tokens
        total_tokens += n_tokens

        if end == len(input_ids):
            break
        start = end - 1

    return total_log_prob / max(total_tokens, 1)


def score_sentences(
    model,
    tokenizer,
    sentences: list[str],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    scores = []
    max_positions = model_max_positions(model)

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(sentences), batch_size), desc="Scoring"):
            raw_batch = sentences[start:start + batch_size]
            batch = [prepare_sentence(tokenizer, sentence) for sentence in raw_batch]
            encoded = tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
            batch_scores: list[float | None] = [None] * len(encoded)

            short_indices = [i for i, ids in enumerate(encoded) if len(ids) <= max_positions]
            long_indices = [i for i, ids in enumerate(encoded) if len(ids) > max_positions]

            if short_indices:
                short_batch = [batch[i] for i in short_indices]
                enc = tokenizer(
                    short_batch,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                for i, score in zip(short_indices, score_batch(model, input_ids, attention_mask)):
                    batch_scores[i] = float(score.item())

            for i in long_indices:
                batch_scores[i] = score_long_sentence(model, encoded[i], device, max_positions)

            scores.extend(score for score in batch_scores if score is not None)

    return scores


def clean_tsv(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def model_parts(model_dir: Path) -> tuple[str, str, str]:
    if model_dir.parent.name.startswith("gpt2-") and model_dir.parents[1].name == "english_baseline":
        return model_dir.parent.name, model_dir.parents[1].name, model_dir.name

    return model_dir.parents[1].name, model_dir.parent.name, model_dir.name


def phenomenon_from_pairs(pairs_path: Path) -> str:
    if pairs_path.parent.name in {"pairs", "english_pairs"}:
        return pairs_path.parent.parent.name

    return pairs_path.parent.name


def output_path_for(pairs_path: Path, model_dir: Path) -> Path:
    model_size, language, seed = model_parts(model_dir)
    phenomenon = phenomenon_from_pairs(pairs_path)

    return (
        Path(RESULTS_DIR)
        / "scoring"
        / model_size
        / seed
        / phenomenon
        / f"{language}.scores.tsv"
    )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs_path = Path(args.pairs)
    model_dir = Path(args.model)
    model_path = model_dir / args.checkpoint
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    output_path = output_path_for(pairs_path, model_dir)
    summary_path = output_path.with_suffix(".summary.json")
    model_size, language, seed = model_parts(model_dir)
    phenomenon = phenomenon_from_pairs(pairs_path)

    rows = read_jsonl(pairs_path)
    goods = []
    bads = []
    for row in rows:
        good = row.get("good")
        bad = row.get("bad")
        if not isinstance(good, str) or not good.strip():
            raise ValueError(f"Missing good sentence: {row}")
        if not isinstance(bad, str) or not bad.strip():
            raise ValueError(f"Missing bad sentence: {row}")
        goods.append(good.strip())
        bads.append(bad.strip())

    device = device_from_arg(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or boundary_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    good_scores = score_sentences(model, tokenizer, goods, device, args.batch_size)
    bad_scores = score_sentences(model, tokenizer, bads, device, args.batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    ties = 0
    delta_sum = 0.0

    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            "pair_index\tid\tlanguage\tphenomenon_id\tgood_score\tbad_score\t"
            "delta\tcorrect\ttie\tgood\tbad\n"
        )
        for index, (row, good_score, bad_score) in enumerate(
            zip(rows, good_scores, bad_scores),
            start=1,
        ):
            delta = good_score - bad_score
            tie = abs(delta) <= 1e-8
            is_correct = delta > 1e-8

            correct += int(is_correct)
            ties += int(tie)
            delta_sum += delta

            f.write(
                f"{row.get('pair_index', index)}\t"
                f"{row.get('id', '')}\t"
                f"{row.get('language', '')}\t"
                f"{row.get('phenomenon_id', '')}\t"
                f"{good_score:.8f}\t{bad_score:.8f}\t{delta:.8f}\t"
                f"{int(is_correct)}\t{int(tie)}\t"
                f"{clean_tsv(row.get('good', ''))}\t"
                f"{clean_tsv(row.get('bad', ''))}\n"
            )

    summary = {
        "model_size": model_size,
        "seed": seed,
        "language": language,
        "phenomenon": phenomenon,
        "model": str(model_dir),
        "checkpoint": str(model_path),
        "pairs": str(pairs_path),
        "output": str(output_path),
        "n_pairs": len(rows),
        "score_mode": SCORE_MODE,
        "accuracy": correct / len(rows),
        "ties": ties,
        "tie_rate": ties / len(rows),
        "mean_delta_good_minus_bad": delta_sum / len(rows),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
