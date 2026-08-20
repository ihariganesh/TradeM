"""
Train/val split for the Plutus fine-tuning dataset.

Stratifies by symbol so the validation set isn't accidentally skewed
toward whichever symbols happen to have more examples — you want val
loss to reflect performance across the whole symbol universe, not just
whichever stock dominates the split by chance.

Usage:
    python finetune/split_dataset.py \
        --input finetune/plutus_finetune_dataset.jsonl \
        --val-fraction 0.12
"""
import argparse
import json
import random
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def get_symbol(example: dict) -> str:
    assistant_content = json.loads(example["messages"][2]["content"])
    return assistant_content.get("symbol", "UNKNOWN")


def stratified_split(examples: list, val_fraction: float, seed: int = 42):
    rng = random.Random(seed)
    by_symbol = defaultdict(list)
    for ex in examples:
        by_symbol[get_symbol(ex)].append(ex)

    train, val = [], []
    for symbol, group in by_symbol.items():
        rng.shuffle(group)
        n_val = (
            max(1, round(len(group) * val_fraction)) if len(group) >= 5 else 0
        )
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_jsonl(args.input)
    train, val = stratified_split(examples, args.val_fraction, args.seed)

    train_path = args.input.replace(".jsonl", "_train.jsonl")
    val_path = args.input.replace(".jsonl", "_val.jsonl")
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    # Report per-symbol breakdown so you can sanity-check the stratification.
    def symbol_counts(rows):
        counts = defaultdict(int)
        for r in rows:
            counts[get_symbol(r)] += 1
        return dict(sorted(counts.items()))

    print(f"Total examples: {len(examples)}")
    print(f"Train: {len(train)} -> {train_path}")
    print(f"Val:   {len(val)} -> {val_path}")
    print(f"\nTrain per-symbol: {symbol_counts(train)}")
    print(f"Val per-symbol:   {symbol_counts(val)}")

    overlap_check = set(json.dumps(e) for e in train) & set(
        json.dumps(e) for e in val
    )
    print(
        f"\nExact-duplicate overlap between train/val: {len(overlap_check)} (should be 0)"
    )
