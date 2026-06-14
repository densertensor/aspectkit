"""Tests for the ASQP, Twitter ($T$), and BIO/ATEPC loaders."""

import pytest

from aspectkit.exceptions import DataFormatError
from aspectkit.io import load_examples, read_asqp, read_bio, read_twitter
from aspectkit.schema import is_implicit

# Format documented in the ABSA-QUAD repository (Zhang et al. 2021):
# sentence####[[aspect, category, polarity, opinion], ...], NULL = implicit.
ASQP = (
    "The wait here is long for dim sum .####"
    "[['wait', 'service general', 'negative', 'long']]\n"
    "this is a sleek alternative .####"
    "[['NULL', 'restaurant miscellaneous', 'negative', 'sleek'], "
    "['alternative', 'restaurant general', 'positive', 'NULL']]\n"
)

# Format of the ACL-14 Twitter dataset (Dong et al. 2014): 3 lines per
# example; polarity -1/0/1.
TWITTER = (
    "i agree about arafat . they even gave one to $T$ ha .\n"
    "jimmy carter\n"
    "-1\n"
    "musicmonday $T$ - lucky do you remember this song ?\n"
    "britney spears\n"
    "1\n"
    "wtf ? $T$ is coming to my school today .\n"
    "hilary swank\n"
    "0\n"
)

# PyABSA ATEPC convention: token TAG polarity; -100 marks unlabelled
# positions; sentences with k aspects are repeated k times.
ATEPC = """I O -100
love O -100
the O -100
cord B-ASP Neutral
and O -100
battery B-ASP -100
life I-ASP -100

I O -100
love O -100
the O -100
cord B-ASP -100
and O -100
battery B-ASP Positive
life I-ASP Positive

Different O -100
sentence O -100
here O -100
"""

PLAIN_BIO = """The O
battery B-ASP
life I-ASP
works O
"""

SUFFIX_BIO = """battery B-POS
life I-POS
but O
screen B-NEG
"""


class TestAsqp:
    @pytest.fixture()
    def path(self, tmp_path):
        p = tmp_path / "train.txt"
        p.write_text(ASQP)
        return p

    def test_explicit_quad(self, path):
        examples = read_asqp(path)
        t = examples[0].tuples[0]
        assert t.aspect.text == "wait"
        assert t.aspect.aligned
        assert examples[0].text[t.aspect.start : t.aspect.end] == "wait"
        assert t.category == "service general"
        assert t.opinion.text == "long"
        assert t.polarity == "negative"

    def test_null_aspect_and_opinion(self, path):
        examples = read_asqp(path)
        implicit_aspect, implicit_opinion = examples[1].tuples
        assert is_implicit(implicit_aspect.aspect)
        assert implicit_aspect.opinion.text == "sleek"
        assert implicit_opinion.aspect.text == "alternative"
        assert is_implicit(implicit_opinion.opinion)

    def test_missing_separator(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("no labels here\n")
        with pytest.raises(DataFormatError, match="separator"):
            read_asqp(p)

    def test_wrong_arity(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("text .####[['a', 'b', 'positive']]\n")
        with pytest.raises(DataFormatError, match="aspect, category, polarity, opinion"):
            read_asqp(p)

    def test_bad_polarity(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("text .####[['a', 'b', 'great', 'c']]\n")
        with pytest.raises(DataFormatError, match="cannot map"):
            read_asqp(p)

    def test_load_examples_alias(self, path):
        assert len(load_examples(path, "asqp")) == 2


class TestTwitter:
    @pytest.fixture()
    def path(self, tmp_path):
        p = tmp_path / "train.raw"
        p.write_text(TWITTER)
        return p

    def test_placeholder_substitution_and_offsets(self, path):
        examples = read_twitter(path)
        assert len(examples) == 3
        first = examples[0]
        assert "$T$" not in first.text
        t = first.tuples[0]
        assert t.aspect.text == "jimmy carter"
        assert first.text[t.aspect.start : t.aspect.end] == "jimmy carter"

    def test_polarity_mapping_is_twitter_convention(self, path):
        # 0 means neutral here, unlike the ACOS integer codes.
        examples = read_twitter(path)
        assert [e.tuples[0].polarity for e in examples] == ["negative", "positive", "neutral"]

    def test_line_count_must_be_multiple_of_three(self, tmp_path):
        p = tmp_path / "bad.raw"
        p.write_text("only $T$ line\naspect\n")
        with pytest.raises(DataFormatError, match="groups of 3"):
            read_twitter(p)

    def test_missing_placeholder(self, tmp_path):
        p = tmp_path / "bad.raw"
        p.write_text("no placeholder\naspect\n1\n")
        with pytest.raises(DataFormatError, match="placeholder"):
            read_twitter(p)

    def test_unknown_polarity_code(self, tmp_path):
        p = tmp_path / "bad.raw"
        p.write_text("with $T$ here\naspect\n2\n")
        with pytest.raises(DataFormatError, match="polarity code"):
            read_twitter(p)

    def test_trailing_blank_lines_tolerated(self, tmp_path):
        p = tmp_path / "ok.raw"
        p.write_text("with $T$ here\naspect\n1\n\n\n")
        assert len(read_twitter(p)) == 1


class TestBio:
    def test_atepc_repeats_merged(self, tmp_path):
        p = tmp_path / "laptops.atepc"
        p.write_text(ATEPC)
        examples = read_bio(p)
        assert len(examples) == 2  # repeated sentence merged, plus the distinct one
        merged = examples[0]
        assert [t.aspect.text for t in merged.tuples] == ["cord", "battery life"]
        assert [t.polarity for t in merged.tuples] == ["neutral", "positive"]
        # offsets point into the space-joined text
        span = merged.tuples[1].aspect
        assert merged.text[span.start : span.end] == "battery life"
        assert examples[1].tuples == []

    def test_repeats_kept_when_disabled(self, tmp_path):
        p = tmp_path / "laptops.atepc"
        p.write_text(ATEPC)
        examples = read_bio(p, merge_repeats=False)
        assert len(examples) == 3
        assert [t.polarity for t in examples[0].tuples] == ["neutral", None]

    def test_plain_two_column_bio(self, tmp_path):
        p = tmp_path / "train.conll"
        p.write_text(PLAIN_BIO)
        (example,) = read_bio(p)
        (t,) = example.tuples
        assert t.aspect.text == "battery life"
        assert t.polarity is None

    def test_polarity_encoded_in_tag_suffix(self, tmp_path):
        p = tmp_path / "e2e.conll"
        p.write_text(SUFFIX_BIO)
        (example,) = read_bio(p)
        assert [(t.aspect.text, t.polarity) for t in example.tuples] == [
            ("battery life", "positive"),
            ("screen", "negative"),
        ]

    def test_adjacent_b_tags_are_separate_spans(self, tmp_path):
        p = tmp_path / "x.conll"
        p.write_text("wifi B-ASP\nscreen B-ASP\n")
        (example,) = read_bio(p)
        assert [t.aspect.text for t in example.tuples] == ["wifi", "screen"]

    def test_orphan_i_tag_tolerated(self, tmp_path):
        p = tmp_path / "x.conll"
        p.write_text("battery I-ASP\nlife I-ASP\n")
        (example,) = read_bio(p)
        assert [t.aspect.text for t in example.tuples] == ["battery life"]

    def test_unknown_tag_rejected(self, tmp_path):
        p = tmp_path / "x.conll"
        p.write_text("battery X-ASP\n")
        with pytest.raises(DataFormatError, match="unrecognised tag"):
            read_bio(p)

    def test_bad_column_count(self, tmp_path):
        p = tmp_path / "x.conll"
        p.write_text("battery B-ASP Positive extra\n")
        with pytest.raises(DataFormatError, match="token TAG"):
            read_bio(p)

    def test_unmappable_polarity_label(self, tmp_path):
        p = tmp_path / "x.conll"
        p.write_text("battery B-ASP Great\n")
        with pytest.raises(DataFormatError, match="cannot map"):
            read_bio(p)

    def test_load_examples_aliases(self, tmp_path):
        p = tmp_path / "x.atepc"
        p.write_text(PLAIN_BIO)
        for name in ("bio", "conll", "atepc"):
            assert len(load_examples(p, name)) == 1
