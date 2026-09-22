"""Word-level vocabularies for the two label spaces used in this project.

The dataset provides two text targets per recording:
  - ``label_no_order``: gesture/ASL-order gloss sequence (e.g. "you from where").
    This is the target for the CTC loss on the encoder output. Its "classes"
    are individual gloss tokens (whitespace-separated words), NOT whole
    sentences, so the model can, in principle, recompose glosses it has seen
    into gloss sequences it has never seen as a whole.
  - ``label_order``: spoken-English-order sentence (e.g. "where are you from").
    This is the target for the decoder's cross-entropy loss. Its "classes"
    are individual English word tokens, again NOT whole sentences, for the
    same open-vocabulary-generalization reason.

Both vocabularies are built ONLY from the train split to avoid leaking
val/test tokens into training, and are then cached to disk so that
validation, test and inference all use the exact same token <-> id mapping.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<pad>"
BLANK_TOKEN = "<blank>"   # CTC blank symbol, gloss vocab only
BOS_TOKEN = "<bos>"       # decoder vocab only
EOS_TOKEN = "<eos>"       # decoder vocab only
UNK_TOKEN = "<unk>"       # both vocabs (defensive, should rarely trigger)


class Vocabulary:
    """A simple token <-> integer-id mapping with a fixed set of specials."""

    def __init__(self, tokens: Iterable[str], specials: list[str]) -> None:
        # Specials always occupy the first indices, in the given order, so
        # that e.g. the CTC blank is reliably index 0.
        self.itos: list[str] = list(specials) + sorted(set(tokens) - set(specials))
        self.stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unk = self.stoi.get(UNK_TOKEN)
        return [self.stoi.get(tok, unk) for tok in tokens]

    def decode(self, ids: Iterable[int], stop_at_eos: bool = False) -> list[str]:
        out = []
        for i in ids:
            tok = self.itos[i]
            if stop_at_eos and tok == EOS_TOKEN:
                break
            out.append(tok)
        return out

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls.__new__(cls)
        vocab.itos = data["itos"]
        vocab.stoi = {tok: idx for idx, tok in enumerate(vocab.itos)}
        return vocab


def _iter_manifest_labels(manifest_path: str | Path, field: str) -> Iterable[list[str]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield record[field].split()


def build_gloss_vocab(train_manifest_path: str | Path) -> Vocabulary:
    """Vocabulary for the CTC target (``label_no_order``). Index 0 == blank."""
    counter: Counter[str] = Counter()
    for tokens in _iter_manifest_labels(train_manifest_path, "label_no_order"):
        counter.update(tokens)
    specials = [BLANK_TOKEN, UNK_TOKEN]
    return Vocabulary(counter.keys(), specials)


def build_word_vocab(train_manifest_path: str | Path) -> Vocabulary:
    """Vocabulary for the decoder target/source (``label_order``)."""
    counter: Counter[str] = Counter()
    for tokens in _iter_manifest_labels(train_manifest_path, "label_order"):
        counter.update(tokens)
    specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    return Vocabulary(counter.keys(), specials)


def build_or_load_vocabs(
    dataset_root: str | Path,
    train_split: str,
    out_dir: str | Path,
    gloss_vocab_file: str,
    word_vocab_file: str,
) -> tuple[Vocabulary, Vocabulary]:
    """Build both vocabularies from the train manifest, or load cached ones."""
    out_dir = Path(out_dir)
    gloss_path = out_dir / gloss_vocab_file
    word_path = out_dir / word_vocab_file

    if gloss_path.is_file() and word_path.is_file():
        return Vocabulary.from_json(gloss_path), Vocabulary.from_json(word_path)

    train_manifest = Path(dataset_root) / "manifests" / f"{train_split}.jsonl"
    gloss_vocab = build_gloss_vocab(train_manifest)
    word_vocab = build_word_vocab(train_manifest)
    gloss_vocab.to_json(gloss_path)
    word_vocab.to_json(word_path)
    return gloss_vocab, word_vocab
