"""Build train/validation/test manifests from a single `all.jsonl` file.

Three split strategies are supported (read from configs/config.yaml under
`data.split`). Each tests a DIFFERENT kind of generalization -- pick the one
that matches what you actually want to measure:

  - "subject"       : keep the original subject-disjoint split, exactly as
                       the `split` field in all.jsonl already assigns it
                       (subject1 -> train, subject2 -> validation, subject3
                       -> test). Tests: does the model work for a SIGNER it
                       has never seen, signing sentences it already knows?

  - "random"        : ignore the `subject`/`split` fields and build a fresh
                       random split across the pooled 7400 recordings, the
                       same style of split the original paper used for its
                       headline results (Section 3.1: "randomly divided
                       into five equal parts"). Stratified by `sentence_id`
                       so every one of the 300 sentences is represented in
                       train. Tests: does the model recognize a NEW
                       RECORDING of a sentence it already knows (same
                       signer pool, same 300 sentences)?

  - "compositional" : holds a subset of ENTIRE sentence_ids out of train
                       completely (all subjects, all recordings of those
                       sentences go to test). The remaining sentences are
                       split at the recording level into train/validation
                       (same idea as "random", but only within the
                       non-held-out sentence pool). Tests: can the model
                       recognize a SENTENCE IT HAS NEVER SEEN, built out of
                       words/glosses it *has* seen elsewhere? This is the
                       closest proxy this dataset supports for "generalizes
                       to similar/new sentences", since the corpus has no
                       real paraphrases -- new word order = the closest
                       thing to a "similar sentence" available here.

                       IMPORTANT: with only 300 sentences and 455-502
                       words, most words are unique to a single sentence
                       (only ~85/300 sentences here contain zero words
                       unique to themselves). Holding out a sentence whose
                       word ONLY appears in that sentence would make that
                       word unlearnable (CTC/word vocab is built from train
                       only). The script greedily picks held-out sentences
                       while checking, after each pick, that no word or
                       gloss would become orphaned (0 remaining sentences
                       containing it) -- so vocabulary coverage in train is
                       always guaranteed. If the requested fraction can't
                       be reached safely, it holds out as many as it safely
                       can and prints a warning.

Usage:
    python build_splits.py --config configs/config.yaml
    python build_splits.py --config configs/config.yaml --split random
    python build_splits.py --config configs/config.yaml --split compositional
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument(
        "--split",
        type=str,
        choices=["subject", "random", "compositional"],
        default=None,
        help="Override data.split from the config file.",
    )
    return p.parse_args()


def load_all_records(all_manifest_path: Path) -> list[dict]:
    with open(all_manifest_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_subject_split(records: list[dict]) -> dict[str, list[dict]]:
    """Trust the `split` field already stored in all.jsonl."""
    out: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for r in records:
        out[r["split"]].append(r)
    return out


def build_random_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Pool everyone together and split randomly, stratified by sentence_id.

    Stratifying means: for EACH of the 300 sentences, its ~20-24 repeated
    recordings (across all subjects) are shuffled and divided in the same
    train/val/test proportions. This guarantees every sentence -- and every
    word in it -- appears in the training set, exactly like the paper's
    random 5-way split of the pooled 7400 recordings.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "train_ratio + val_ratio + test_ratio must sum to 1.0"
    )
    rng = random.Random(seed)

    by_sentence: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_sentence[r["sentence_id"]].append(r)

    out: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for sentence_id, group in by_sentence.items():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        # whatever remains goes to test, so rounding never drops a record
        n_test = n - n_train - n_val

        out["train"].extend(group[:n_train])
        out["validation"].extend(group[n_train : n_train + n_val])
        out["test"].extend(group[n_train + n_val :])

    # Shuffle each split's overall order too (otherwise records are grouped
    # by sentence_id, which is harmless for training but tidier to avoid).
    for key in out:
        rng.shuffle(out[key])
    return out


def _label_tokens(record: dict) -> tuple[list[str], list[str]]:
    """Returns (decoder_words, gloss_words) for one recording's labels."""
    return record["label_order"].split(), record["label_no_order"].split()


def build_compositional_split(
    records: list[dict],
    unseen_sentence_fraction: float,
    train_recording_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Hold whole sentences out of train to test generalization to unseen
    sentences built from already-seen words/glosses.

    Step 1: greedily pick `unseen_sentence_fraction` of the 300 sentence_ids
            to fully exclude from train, verifying after each pick that no
            decoder word or gloss becomes "orphaned" (present in zero of
            the remaining, non-held-out sentences). ALL recordings (all
            subjects) of a held-out sentence go to `test`.
    Step 2: for every sentence_id NOT held out, split its recordings at the
            recording level (stratified per sentence, mixing subjects)
            into train / validation using `train_recording_ratio`. This
            half of the split is the same idea as build_random_split, just
            restricted to the "safe" sentence pool.
    """
    rng = random.Random(seed)

    by_sentence: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_sentence[r["sentence_id"]].append(r)

    word_to_sents: dict[str, set[int]] = defaultdict(set)
    gloss_to_sents: dict[str, set[int]] = defaultdict(set)
    for sid, group in by_sentence.items():
        words, glosses = _label_tokens(group[0])
        for w in words:
            word_to_sents[w].add(sid)
        for g in glosses:
            gloss_to_sents[g].add(sid)

    all_sids = list(by_sentence.keys())
    rng.shuffle(all_sids)

    target = round(len(all_sids) * unseen_sentence_fraction)
    held_out: set[int] = set()

    def safe_to_holdout(sid: int) -> bool:
        trial = held_out | {sid}
        words, glosses = _label_tokens(by_sentence[sid][0])
        for w in words:
            if word_to_sents[w] - trial == set():
                return False
        for g in glosses:
            if gloss_to_sents[g] - trial == set():
                return False
        return True

    for sid in all_sids:
        if len(held_out) >= target:
            break
        if safe_to_holdout(sid):
            held_out.add(sid)

    if len(held_out) < target:
        print(
            f"[warn] requested unseen_sentence_fraction={unseen_sentence_fraction} "
            f"({target} sentences) but only {len(held_out)} could be held out "
            f"safely without orphaning a word/gloss from train's vocabulary. "
            f"Continuing with {len(held_out)}."
        )

    # Sanity check: every word/gloss must still be covered by the training pool.
    train_pool_sids = set(all_sids) - held_out
    for w, sset in word_to_sents.items():
        assert sset - held_out, f"orphaned decoder word after holdout: {w!r}"
    for g, sset in gloss_to_sents.items():
        assert sset - held_out, f"orphaned gloss after holdout: {g!r}"

    out: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}

    # All recordings of held-out sentences -> test (novel-sentence generalization).
    for sid in held_out:
        out["test"].extend(by_sentence[sid])

    # Remaining sentences: split recordings into train/validation, stratified.
    for sid in train_pool_sids:
        group = list(by_sentence[sid])
        rng.shuffle(group)
        n_train = round(len(group) * train_recording_ratio)
        out["train"].extend(group[:n_train])
        out["validation"].extend(group[n_train:])

    for key in out:
        rng.shuffle(out[key])

    print(
        f"[info] compositional split: {len(held_out)}/{len(all_sids)} sentences "
        f"held out entirely -> test; {len(train_pool_sids)} sentences split "
        f"recording-wise into train/validation"
    )
    return out


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg["data"]["dataset_root"])
    split_cfg = cfg["data"].get("split", {})
    split_mode = args.split or split_cfg.get("mode", "subject")

    all_manifest_name = split_cfg.get("all_manifest", "manifests/all.jsonl")
    all_manifest_path = dataset_root / all_manifest_name
    records = load_all_records(all_manifest_path)
    print(f"[info] loaded {len(records)} records from {all_manifest_path}")

    if split_mode == "subject":
        splits = build_subject_split(records)
        print("[info] using SUBJECT-DISJOINT split (subject1->train, subject2->val, subject3->test)")
    elif split_mode == "random":
        ratios = split_cfg.get("random_ratios", {"train": 0.8108, "validation": 0.1351, "test": 0.0541})
        seed = split_cfg.get("random_seed", 42)
        splits = build_random_split(
            records,
            train_ratio=ratios["train"],
            val_ratio=ratios["validation"],
            test_ratio=ratios["test"],
            seed=seed,
        )
        print(f"[info] using RANDOM split (stratified by sentence_id, seed={seed})")
    elif split_mode == "compositional":
        comp_cfg = split_cfg.get("compositional", {})
        unseen_fraction = comp_cfg.get("unseen_sentence_fraction", 0.15)
        train_recording_ratio = comp_cfg.get("train_recording_ratio", 0.85)
        seed = comp_cfg.get("seed", split_cfg.get("random_seed", 42))
        splits = build_compositional_split(
            records,
            unseen_sentence_fraction=unseen_fraction,
            train_recording_ratio=train_recording_ratio,
            seed=seed,
        )
        print("[info] using COMPOSITIONAL split (test = entirely unseen sentences built from known words)")
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    for name, key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        out_path = dataset_root / "manifests" / f"{name}.jsonl"
        write_manifest(splits[key], out_path)
        print(f"[info] wrote {len(splits[key])} records -> {out_path}")


if __name__ == "__main__":
    main()