#!/usr/bin/env python3
"""Tests for lab 1-1. Usage: ./test_1_1.py [todo_number]"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

LAB_PATH = Path(__file__).resolve().parent / "lab-1-1-tokenizers.py"

TOY_TEXTS = [
    "low low low low low",
    "lower lower widest widest widest",
    "newest newest newest newest newest newest",
]


def load_lab():
    spec = importlib.util.spec_from_file_location("lab", LAB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab"] = module
    spec.loader.exec_module(module)
    return module


def check(name, got_fn, expected, compare=None):
    """Run got_fn(), compare against expected, print a PASS/FAIL line."""
    try:
        got = got_fn()
    except NotImplementedError:
        print(f"  [FAIL] {name}\n         not implemented yet")
        return False
    except Exception as error:
        print(f"  [FAIL] {name}\n         raised {type(error).__name__}: {error}")
        return False
    ok = compare(got, expected) if compare else got == expected
    if ok:
        print(f"  [PASS] {name}")
    else:
        print(
            f"  [FAIL] {name}\n         expected {expected!r}\n         got      {got!r}"
        )
    return ok


def check_raises(name, fn, exc_type, message_substring):
    try:
        fn()
    except exc_type as error:
        if message_substring in str(error):
            print(f"  [PASS] {name}")
            return True
        print(
            f"  [FAIL] {name}\n         expected message containing {message_substring!r}\n         got      {str(error)!r}"
        )
        return False
    except Exception as error:
        print(
            f"  [FAIL] {name}\n         expected {exc_type.__name__}, got {type(error).__name__}: {error}"
        )
        return False
    print(f"  [FAIL] {name}\n         expected {exc_type.__name__}, nothing raised")
    return False


def trained_tokenizer(lab, texts, vocab_size):
    """Train a tokenizer. Needs TODOs 2-6; raises NotImplementedError otherwise."""
    tokenizer = lab.SimpleBPETokenizer()
    tokenizer.train(texts, vocab_size=vocab_size)
    return tokenizer


def trained_toy_tokenizer(lab, extra_pieces=6):
    return trained_tokenizer(lab, TOY_TEXTS, 256 + extra_pieces)


def todo_1(lab):
    """TODO 1: manual_pretokenization, manual_toy_bpe and manual_encode"""
    return [
        check(
            "one pre-token list per example",
            lambda: len(lab.manual_pretokenization()),
            4,
        ),
        check(
            "a leading space stays with the word after it",
            lambda: lab.manual_pretokenization()[0][1],
            "Ġcat",
        ),
        check(
            "the second example has five pre-tokens",
            lambda: len(lab.manual_pretokenization()[1]),
            5,
        ),
        check(
            "the multilingual example has four pre-tokens",
            lambda: len(lab.manual_pretokenization()[3]),
            4,
        ),
        check(
            "six single-byte pieces plus four merged pieces",
            lambda: len(lab.manual_toy_bpe()),
            10,
        ),
        check("the first merged piece", lambda: lab.manual_toy_bpe()[6], "at"),
        check(
            "the last piece is four characters long",
            lambda: len(lab.manual_toy_bpe()[9]),
            4,
        ),
        check(
            "encoding the example text by hand",
            lambda: lab.manual_encode(),
            [0, 5, 1],
        ),
    ]


def todo_2(lab):
    """TODO 2: count_pretokens"""
    return [
        check(
            "counts repeated pre-tokens with their leading space",
            lambda: lab.count_pretokens(["low low"]),
            Counter({(b"l", b"o", b"w"): 1, (b" ", b"l", b"o", b"w"): 1}),
        ),
        check(
            "counts across several texts",
            lambda: lab.count_pretokens(["low lower", "low"]),
            Counter({(b"l", b"o", b"w"): 2, (b" ", b"l", b"o", b"w", b"e", b"r"): 1}),
        ),
        check(
            "splits non-ascii text into single bytes",
            lambda: lab.count_pretokens(["hi 😊"]),
            Counter({(b"h", b"i"): 1, (b" ", b"\xf0", b"\x9f", b"\x98", b"\x8a"): 1}),
        ),
        check(
            "no texts gives an empty counter",
            lambda: lab.count_pretokens([]),
            Counter(),
        ),
        check(
            "whitespace-only text gives an empty counter",
            lambda: lab.count_pretokens(["   "]),
            Counter(),
        ),
    ]


def todo_3(lab):
    """TODO 3: get_pair_counts"""
    token_counts = Counter(
        {(b"l", b"o", b"w"): 3, (b" ", b"l", b"o", b"w"): 2, (b"x",): 5}
    )
    return [
        check(
            "sums pair counts over pre-tokens",
            lambda: lab.get_pair_counts(token_counts),
            Counter({(b"l", b"o"): 5, (b"o", b"w"): 5, (b" ", b"l"): 2}),
        ),
        check(
            "a single-piece pre-token has no pairs",
            lambda: lab.get_pair_counts(Counter({(b"a",): 7})),
            Counter(),
        ),
        check(
            "empty input gives an empty counter",
            lambda: lab.get_pair_counts(Counter()),
            Counter(),
        ),
    ]


def todo_4(lab):
    """TODO 4: merge_pair_in_token"""
    return [
        check(
            "merges a single occurrence",
            lambda: lab.merge_pair_in_token(
                (b"l", b"o", b"w", b"e", b"r"), (b"l", b"o")
            ),
            (b"lo", b"w", b"e", b"r"),
        ),
        check(
            "overlapping occurrences are consumed left to right",
            lambda: lab.merge_pair_in_token((b"a", b"a", b"a"), (b"a", b"a")),
            (b"aa", b"a"),
        ),
        check(
            "merges two non-overlapping occurrences",
            lambda: lab.merge_pair_in_token((b"a", b"a", b"a", b"a"), (b"a", b"a")),
            (b"aa", b"aa"),
        ),
        check(
            "absent pair leaves the token unchanged",
            lambda: lab.merge_pair_in_token((b"l", b"o", b"w"), (b"x", b"y")),
            (b"l", b"o", b"w"),
        ),
        check(
            "single-piece token is unchanged",
            lambda: lab.merge_pair_in_token((b"a",), (b"a", b"a")),
            (b"a",),
        ),
        check(
            "empty token stays empty",
            lambda: lab.merge_pair_in_token((), (b"a", b"a")),
            (),
        ),
    ]


def todo_5(lab):
    """TODO 5: merge_pair"""
    token_counts = Counter({(b"l", b"o", b"w"): 3, (b"lo", b"w"): 2, (b"n", b"o"): 1})
    return [
        check(
            "counts of pre-tokens that become identical are summed",
            lambda: lab.merge_pair(token_counts, (b"l", b"o")),
            Counter({(b"lo", b"w"): 5, (b"n", b"o"): 1}),
        ),
        check(
            "absent pair leaves the counter unchanged",
            lambda: lab.merge_pair(Counter({(b"a", b"b"): 4}), (b"x", b"y")),
            Counter({(b"a", b"b"): 4}),
        ),
        check(
            "empty counter stays empty",
            lambda: lab.merge_pair(Counter(), (b"a", b"b")),
            Counter(),
        ),
    ]


def todo_6(lab):
    """TODO 6: SimpleBPETokenizer.train"""
    return [
        check(
            "learns the expected merges on the toy corpus",
            lambda: trained_toy_tokenizer(lab).merges,
            [
                (b"s", b"t"),
                (b"e", b"st"),
                (b"o", b"w"),
                (b"l", b"ow"),
                (b"w", b"est"),
                (b"n", b"e"),
            ],
        ),
        check(
            "adds one vocabulary piece per merge",
            lambda: [
                trained_toy_tokenizer(lab).id_to_piece[i] for i in range(256, 262)
            ],
            [b"st", b"est", b"ow", b"low", b"west", b"ne"],
        ),
        check(
            "vocabulary grows to the requested size",
            lambda: trained_toy_tokenizer(lab, extra_pieces=2).vocab_size,
            258,
        ),
        check(
            "stops early when no pairs are left",
            lambda: trained_tokenizer(lab, ["a a"], 300).merges,
            [(b" ", b"a")],
        ),
    ]


def todo_7(lab):
    """TODO 7: SimpleBPETokenizer.encode"""
    return [
        check(
            "untrained tokenizer encodes raw utf-8 bytes",
            lambda: lab.SimpleBPETokenizer().encode("hi"),
            [104, 105],
        ),
        check(
            "untrained tokenizer encodes non-ascii per byte",
            lambda: lab.SimpleBPETokenizer().encode("é"),
            [195, 169],
        ),
        check(
            "empty text encodes to no ids",
            lambda: lab.SimpleBPETokenizer().encode(""),
            [],
        ),
        check(
            "a fully merged pre-token becomes one id",
            lambda: trained_toy_tokenizer(lab).encode("low"),
            [259],
        ),
        check(
            "merges apply inside each pre-token only",
            lambda: trained_toy_tokenizer(lab).encode("lower newest"),
            [259, 101, 114, 32, 261, 260],
        ),
    ]


def todo_8(lab):
    """TODO 8: SimpleBPETokenizer.decode"""
    return [
        check(
            "decodes single-byte ids",
            lambda: lab.SimpleBPETokenizer().decode([104, 105]),
            "hi",
        ),
        check(
            "decodes a multi-byte utf-8 character",
            lambda: lab.SimpleBPETokenizer().decode([0xC3, 0xA9]),
            "é",
        ),
        check(
            "no ids decodes to the empty string",
            lambda: lab.SimpleBPETokenizer().decode([]),
            "",
        ),
        check(
            "invalid utf-8 becomes the replacement character",
            lambda: lab.SimpleBPETokenizer().decode([0xFF]),
            "�",
        ),
        check(
            "round trips through a trained tokenizer",
            lambda: trained_toy_tokenizer(lab).decode(
                trained_toy_tokenizer(lab).encode("lower newest")
            ),
            "lower newest",
        ),
    ]


TODOS = {
    1: ("manual_pretokenization, manual_toy_bpe, manual_encode", todo_1),
    2: ("count_pretokens", todo_2),
    3: ("get_pair_counts", todo_3),
    4: ("merge_pair_in_token", todo_4),
    5: ("merge_pair", todo_5),
    6: ("SimpleBPETokenizer.train", todo_6),
    7: ("SimpleBPETokenizer.encode", todo_7),
    8: ("SimpleBPETokenizer.decode", todo_8),
}


def main():
    requested = sys.argv[1:]
    numbers = [int(n) for n in requested] if requested else sorted(TODOS)
    lab = load_lab()
    passed = total = 0
    for number in numbers:
        title, runner = TODOS[number]
        print(f"TODO {number}: {title}")
        results = runner(lab)
        passed += sum(results)
        total += len(results)
        print()
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
