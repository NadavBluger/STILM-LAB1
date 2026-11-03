"""Lab 1-1: Byte-pair encoding tokenizers.

In this lab you build a small byte-level BPE tokenizer from scratch. The
implementation is intentionally simple: it is meant to show the algorithm
clearly, not to compete with optimized tokenizer libraries.

Later language-modeling labs will add batching and special-token conventions
around this BPE core.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from dataclasses import dataclass

from datasets import load_dataset
from tqdm import tqdm


@dataclass
class Config:
    dataset_name: str = "roneneldan/TinyStories"
    max_train_examples: int = 500
    vocab_size: int = 1_000
    compare_vocab_sizes: str = "300,1000,2000"
    max_examples_to_print: int = 5
    embedding_size: int = 512


def bpe_pretokenize(text: str) -> list[str]:
    """Split a text into pre-tokens, before any BPE merge is applied.

    A pre-token is a word or a single punctuation character, together with the
    whitespace in front of it. Keeping a leading space with the following token
    makes word boundaries visible to BPE without learning a separate token for
    every word+punctuation variant. Whitespace at the end of the text, with no
    word or punctuation after it, is dropped.

    Args:
        text: the text to split.

    Returns:
        The pre-tokens of the text, in order.

    Example:
        bpe_pretokenize("the cat sat.") -> ["the", " cat", " sat", "."]
    """
    return re.findall(r"\s*\w+|\s*[^\w\s]", text)


def mark_spaces(text: str) -> str:
    return text.replace(" ", "Ġ")


def manual_pretokenization() -> list[list[str]]:
    """The BPE pre-tokens of each string in `texts`.

    Work the answer out by hand and write it down as a literal.
    Note: A space kept inside a pre-token is written as "Ġ", the way `mark_spaces` writes it.

    Returns:
        One list of pre-tokens per example, in the order of the examples.

    Example:
        for the examples ["go now", "hi!"], the answer is
            [["go", "Ġnow"], ["hi", "!"]]
    """
    texts = [
        "the cat and the theater",
        "Once upon a time.",
        "I'll go there!",
        "What a funny character 😊",
    ]
    # TODO COMPLETE (1)
    return [["the", "Ġcat", "Ġand", "Ġthe", "Ġtheater"],
            ["Once", "Ġupon", "Ġa", "Ġtime", "."],
            ["I'll", "Ġgo", "Ġthere", "!"],
            ["What", "Ġa", "Ġfunny", "Ġcharacter", "Ġ😊"]]
  


def manual_toy_bpe() -> list[str]:
    """
    Perform BPE to generate a vocabulary of size 10, trained on the corups 'texts'.

    Work the answer out by hand and write it down as a literal.
    Every merge joins the
    most frequent adjacent pair of pieces in the corpus, counting each
    pre-token as often as it occurs. Ties go to the lexicographically largest
    pair. Pieces are written the way `piece_to_text` writes them, so a space is
    "Ġ".

    The vocabulary holds every single-byte piece that occurs in the corpus,
    ordered by byte value, followed by the piece each merge creates, in the
    order the merges are learned. The corpus is made of 6 distinct bytes and
    four merges are learned, so the vocabulary has 6 + 4 = 10 pieces.

    Returns:
        The 10 pieces of the vocabulary, in vocabulary order (space is first,
        then single-byte tokens, then the ordered merge results).

    Example:
        for the corpus ["ab ab"] and two merges, the vocabulary is
            ["Ġ", "a", "b", "ab", "Ġab"]
    """
    texts = ["cat cat cat", "hat hat", "cap"]
    return ["Ġ", "a", "c", "h", "p", "t", "at", "cat", "hat", "Ġcat"]


def manual_encode() -> list[int]:
    """Encode `text` into token IDs with the BPE `vocabulary` and `merges` below.

    Work out the answer by hand and write it as a literal.
    The vocabulary and the merges are given.
    They are not related to the corpora of the other manual functions.

    Encoding follows these steps:
        1. Split `text` into pre-tokens.
        2. Split each pre-token into single characters.
        3. Apply the merges in the order of `merges`. Apply a merge at every place where its pair occurs. Then go to the next merge.
        4. Replace each piece with its ID in `vocabulary`.

    Returns:
        The token IDs of `text`, in order.

    Example:
        For the text "ab", the vocabulary {"a": 0, "b": 1, "ab": 2} and the
        merges [("a", "b")], the answer is [2].
    """
    text = "a cab"
    vocabulary = {"a": 0, "b": 1, "c": 2, "Ġ": 3, "ca": 4, "Ġca": 5, "ab": 6}
    merges = [("c", "a"), ("Ġ", "ca"), ("a", "b")]
    return [0, 5,1]
    

def show_special_tokens() -> None:
    print("# Special tokens")
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    print("special tokens:", special_tokens)
    print("[BOS] and [EOS] mark sequence boundaries for language modeling.")
    print("[PAD] is useful when batching variable-length sequences.")
    print(
        "[UNK] is mostly unnecessary for byte-level BPE, but common in tokenizer APIs."
    )
    print("In this lab we keep them separate from the BPE algorithm itself.")
    print()


def show_encoding_examples(
    tokenizer: SimpleBPETokenizer, texts: list[str], max_examples: int
) -> None:
    print("# Encoding examples")
    examples = [
        "Once upon a time there was a little girl.",
        "שלום, سلام 😊",
    ]
    examples.extend(texts[:max_examples])

    for text in examples[: max_examples + 2]:
        token_ids = tokenizer.encode(text)
        print(repr(text[:120]))
        print("ids:   ", token_ids[:40])
        print("tokens:", tokens_as_text(tokenizer, token_ids[:40]))
        print("decode matches:", tokenizer.decode(token_ids) == text)
        print()


def str_to_bytes(text: str) -> tuple[bytes, ...]:
    """Split a text into its UTF-8 bytes, one single-byte piece per byte.

    Example:
        str_to_bytes("hé") -> (b"h", b"\\xc3", b"\\xa9")
    """
    return tuple(bytes([byte_value]) for byte_value in text.encode("utf-8"))


def piece_to_text(piece: bytes) -> str:
    """Write a piece as readable text, with each space written as "Ġ".

    A piece that is not valid UTF-8 is written as its hex byte values.

    Example:
        piece_to_text(b" cat") -> "Ġcat"
        piece_to_text(b"\\xf0\\x9f") -> "<bytes f0 9f>"
    """
    try:
        return mark_spaces(piece.decode("utf-8"))
    except UnicodeDecodeError:
        byte_values = " ".join(f"{byte_value:02x}" for byte_value in piece)
        return f"<bytes {byte_values}>"


def count_pretokens(texts: list[str]) -> Counter:
    """Count the pre-tokens of a corpus, as tuples of single-byte pieces.
       i.e. each pre-token is split to its single-byte pieces (characters or bytes in UTF-8),
       and the counts of each pre-token (as a tuple) are returned.

    Args:
        texts: the corpus, one string per document.

    Returns:
        A Counter mapping a tuple of single-byte pieces to the number of times
        that pre-token occurs in the corpus. Empty pre-tokens are not counted.

    Example:
        count_pretokens(["hi hi", "hi"]) ->
            Counter({(b"h", b"i"): 2, (b" ", b"h", b"i"): 1})
    """
    pre_toknized = []
    for text in texts:
        pre_tokens = bpe_pretokenize(text)
        bytes_ = tuple(str_to_bytes(pre_token) for pre_token in pre_tokens)
        pre_toknized.extend(bytes_)
    counts = Counter(pre_toknized)
    return counts


def get_pair_counts(token_counts: Counter) -> Counter:
    """Count the adjacent piece pairs of the counted pre-tokens.

    Args:
        token_counts: pre-token counts, as returned by `count_pretokens`.

    Returns:
        A Counter mapping a `(left, right)` pair of adjacent pieces to how often
        that pair occurs in the corpus.

    Example:
        get_pair_counts(Counter({(b"l", b"o", b"w"): 3, (b" ", b"l", b"o", b"w"): 1})) ->
            Counter({(b'l', b'o'): 4, (b'o', b'w'): 4, (b' ', b'l'): 1})
    """
    pairs_list = []
    for key, val in token_counts.items():
        for key_pair in zip(key[:-1], key[1:]):  # key is tuple
            pairs_list += [key_pair] * val

    pair_counts = Counter(pairs_list)
    return pair_counts



def merge_pair_in_token(
    token: tuple[bytes, ...], pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    """Merge every occurrence of `pair` inside a single pre-token.

    Occurrences are merged greedily from left to right, so two overlapping
    occurrences cannot both survive.

    Args:
        token: a pre-token, as a tuple of pieces.
        pair: the `(left, right)` pieces to join into one piece.

    Returns:
        The pre-token with every occurrence of `pair` replaced by `left + right`.

    Example:
        merge_pair_in_token((b"l", b"o", b"w", b"e", b"r"), (b"l", b"o")) ->
            (b"lo", b"w", b"e", b"r")
        merge_pair_in_token((b"a", b"a", b"a"), (b"a", b"a")) -> (b"aa", b"a")
    """
    merged: list[bytes] = []
    # TODO COMPLETE (4)
    raise NotImplementedError("TODO 4")
    return tuple(merged)


def merge_pair(token_counts: Counter, pair: tuple[bytes, bytes]) -> Counter:
    """Merge every occurrence of `pair` in a whole corpus of counted pre-tokens.

    Args:
        token_counts: pre-token counts, as returned by `count_pretokens`.
        pair: the `(left, right)` pieces to join into one piece.

    Returns:
        A new Counter over the merged pre-tokens, with the same total count.

    Example:
        merge_pair(Counter({(b"l", b"o", b"w"): 3, (b" ", b"l", b"o", b"w"): 1}),
                   (b"l", b"o")) -> Counter({(b"lo", b"w"): 3, (b" ", b"lo", b"w"): 1})
    """
    new_counts = Counter()
    # TODO COMPLETE (5)
    raise NotImplementedError("TODO 5")
    return new_counts


class SimpleBPETokenizer:
    """A small byte-level BPE tokenizer for teaching.

    The vocabulary maps token IDs to byte strings. Text is first pre-tokenized,
    each pre-token is split into UTF-8 bytes, and then learned BPE merges are
    applied inside each pre-token.
    """

    def __init__(self):
        self.id_to_piece: dict[int, bytes] = {}
        self.piece_to_id: dict[bytes, int] = {}
        self.merges: list[tuple[bytes, bytes]] = []

        for byte_value in range(256):
            self.add_piece(bytes([byte_value]))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_piece)

    def add_piece(self, piece: bytes) -> int:
        """Add a piece to the vocabulary and return its token ID.

        A new piece gets the next free ID. A piece that is already in the
        vocabulary keeps its ID, and the vocabulary does not change.
        """
        if piece in self.piece_to_id:
            return self.piece_to_id[piece]
        token_id = len(self.id_to_piece)
        self.id_to_piece[token_id] = piece
        self.piece_to_id[piece] = token_id
        return token_id

    def train(self, texts: list[str], vocab_size: int) -> None:
        """Learn BPE merges from a corpus until the vocabulary reaches `vocab_size`.

        Each merge joins the most frequent adjacent pair of pieces in the corpus
        and adds the joined piece to the vocabulary. Ties between equally
        frequent pairs go to the lexicographically largest pair. Stops early if
        no pair is left to merge.

        Fills in `self.merges`, in the order the merges were learned, and the
        vocabulary. Returns nothing.

        Args:
            texts: the training corpus, one string per document.
            vocab_size: the size the vocabulary should reach. The 256 single
                bytes are already in it, so at most `vocab_size - 256` merges
                are learned. Must be at least 256.

        Example:
            train(["ab ab ab"], vocab_size=258) leaves
                self.merges == [(b"a", b"b"), (b" ", b"ab")]
        """
        if vocab_size < self.vocab_size:
            raise ValueError(f"vocab_size must be at least {self.vocab_size}")

        # TODO COMPLETE (6)
        raise NotImplementedError("TODO 6")

    def encode(self, text: str) -> list[int]:
        """Turn text into token IDs.

        Merges are applied inside each pre-token only, never across a pre-token
        boundary.

        Args:
            text: the text to encode.

        Returns:
            The IDs of the pieces the text is made of, in order.

        Example:
            SimpleBPETokenizer().encode("hi") -> [104, 105]
        """
        token_ids: list[int] = []
        # TODO COMPLETE (7)
        raise NotImplementedError("TODO 7")
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Turn token IDs back into text.

        An arbitrary list of IDs need not spell out valid UTF-8, since a piece
        can hold half of a multi-byte character. Invalid byte sequences become
        the replacement character instead of raising.

        Args:
            token_ids: the IDs to decode.

        Returns:
            The decoded text.

        Example:
            SimpleBPETokenizer().decode([104, 105]) -> "hi"
        """
        # TODO COMPLETE (8)
        raise NotImplementedError("TODO 8")


def longest_tokens(
    tokenizer: SimpleBPETokenizer, n: int = 10
) -> list[tuple[int, str, int]]:
    """Return the `n` longest pieces of the vocabulary, longest first.

    Each item is (token ID, piece as text, length of the piece in bytes).

    Example:
        for a tokenizer with ID 256 = b"ca" and ID 257 = b" cat",
        longest_tokens(tokenizer, n=2) -> [(257, "Ġcat", 4), (256, "ca", 2)]
    """
    pieces = list(tokenizer.id_to_piece.items())
    pieces.sort(key=lambda item: (len(item[1]), item[1]), reverse=True)
    return [
        (token_id, piece_to_text(piece), len(piece)) for token_id, piece in pieces[:n]
    ]


def tokens_as_text(tokenizer: SimpleBPETokenizer, token_ids: list[int]) -> list[str]:
    """Write each token ID as the text of its piece, in the style of `piece_to_text`.

    Example:
        for a tokenizer with ID 257 = b" cat",
        tokens_as_text(tokenizer, [104, 257]) -> ["h", "Ġcat"]
    """
    tokens: list[str] = []
    for token_id in token_ids:
        piece = tokenizer.id_to_piece[token_id]
        tokens.append(piece_to_text(piece))
    return tokens


def load_tinystories_texts(config: Config) -> list[str]:
    dataset = load_dataset(
        config.dataset_name, split=f"train[:{config.max_train_examples}]"
    )
    return list(dataset["text"])


def bytes_per_token(texts: list[str], tokenizer: SimpleBPETokenizer) -> float:
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
    return total_bytes / total_tokens


def estimate_embedding_and_output_parameters(
    vocab_size: int, embedding_size: int
) -> int:
    return 2 * vocab_size * embedding_size


def train_and_report(
    texts: list[str], vocab_size: int, config: Config
) -> SimpleBPETokenizer:
    tokenizer = SimpleBPETokenizer()
    start_time = time.perf_counter()
    tokenizer.train(texts, vocab_size=vocab_size)
    train_seconds = time.perf_counter() - start_time

    ratio = bytes_per_token(texts, tokenizer)
    params = estimate_embedding_and_output_parameters(
        tokenizer.vocab_size, config.embedding_size
    )
    print(f"vocab size: {tokenizer.vocab_size:,}")
    print(f"merges learned: {len(tokenizer.merges):,}")
    print(f"training time: {train_seconds:.2f} seconds")
    print(f"bytes per token on training subset: {ratio:.2f}")
    print(f"embedding + output parameters at d={config.embedding_size}: {params:,}")
    print("longest learned tokens:")
    for token_id, token_text, num_bytes in longest_tokens(tokenizer, n=8):
        print(f"  id={token_id:4d} bytes={num_bytes:2d} token={token_text!r}")
    print()
    return tokenizer


def compare_with_hf_tokenizer(
    texts: list[str],
    tokenizer: SimpleBPETokenizer,
    vocab_size: int,
) -> None:
    print("# Comparison with Hugging Face tokenizers")
    # Note: the from-scratch tokenizer is intentionally simple and recomputes
    # more work than necessary. The speed gap with HF is worth a closer look.
    # The optional extension is to investigate how the optimized implementation
    # gets its speed. Possible directions: cache pair counts,
    # update only counts affected by the latest merge, store merges in a faster
    # lookup structure during encoding, and avoid repeatedly scanning every
    # pre-token from the beginning.
    try:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError:
        print("Install tokenizers to run the Hugging Face comparison.")
        print()
        return

    hf_tokenizer = HFTokenizer(BPE())
    hf_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    hf_tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=ByteLevel.alphabet(),
    )

    start_time = time.perf_counter()
    hf_tokenizer.train_from_iterator(texts, trainer=trainer)
    hf_train_seconds = time.perf_counter() - start_time

    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    our_total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
    hf_total_tokens = sum(len(hf_tokenizer.encode(text).ids) for text in texts)

    print(f"from-scratch bytes/token: {total_bytes / our_total_tokens:.2f}")
    print(f"HF bytes/token:           {total_bytes / hf_total_tokens:.2f}")
    print(f"HF training time:         {hf_train_seconds:.2f} seconds")

    example = "Once upon a time there was a little girl."
    print(repr(example))
    print("from scratch:", tokens_as_text(tokenizer, tokenizer.encode(example)))
    print("HF:          ", hf_tokenizer.encode(example).tokens)

    multilingual = "שלום, سلام 😊"
    hf_encoding = hf_tokenizer.encode(multilingual)
    print(repr(multilingual))
    print(
        "from scratch round trip:",
        tokenizer.decode(tokenizer.encode(multilingual)) == multilingual,
    )
    print(
        "HF round trip:          ", hf_tokenizer.decode(hf_encoding.ids) == multilingual
    )
    print()


def parse_vocab_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small byte-level BPE tokenizer."
    )
    parser.add_argument(
        "--max-train-examples", type=int, default=Config.max_train_examples
    )
    parser.add_argument("--vocab-size", type=int, default=Config.vocab_size)
    parser.add_argument(
        "--compare-vocab-sizes", type=str, default=Config.compare_vocab_sizes
    )
    parser.add_argument(
        "--max-examples-to-print", type=int, default=Config.max_examples_to_print
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        max_train_examples=args.max_train_examples,
        vocab_size=args.vocab_size,
        compare_vocab_sizes=args.compare_vocab_sizes,
        max_examples_to_print=args.max_examples_to_print,
    )
    print("# Training BPE on a TinyStories subset")
    texts = load_tinystories_texts(config)
    print(f"training examples: {len(texts):,}")
    print()

    tokenizer = train_and_report(texts, config.vocab_size, config)
    show_special_tokens()
    show_encoding_examples(tokenizer, texts, config.max_examples_to_print)

    print("# Vocabulary-size tradeoff")
    for vocab_size in parse_vocab_sizes(config.compare_vocab_sizes):
        print(f"Training tokenizer with vocab_size={vocab_size:,}")
        train_and_report(texts, vocab_size, config)

    print("# BPE core used by later labs")
    print("vocab_size:", tokenizer.vocab_size)
    print("merges:    ", len(tokenizer.merges))
    print(
        "encode/decode round trip:",
        tokenizer.decode(tokenizer.encode("Once upon a time")),
    )
    print()

    compare_with_hf_tokenizer(texts, tokenizer, config.vocab_size)


if __name__ == "__main__":
    #main()
    #print(count_pretokens(["hi hi", "hi"]))
    tmp = get_pair_counts(Counter({(b"l", b"o", b"w"): 3, (b" ", b"l", b"o", b"w"): 1}))
    print(tmp)
