"""Tests for the dataset loaders, on faithful miniature fixtures."""

import pytest

from aspectkit.exceptions import DataFormatError
from aspectkit.io import (
    load_examples,
    read_acos,
    read_aste,
    read_jsonl,
    read_semeval_2014,
    read_semeval_2015,
    write_jsonl,
)
from aspectkit.schema import IMPLICIT, ABSAExample, SentimentTuple, Span, is_implicit

SEMEVAL_2014 = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
  <sentence id="813">
    <text>All the appetizers and salads were fabulous!</text>
    <aspectTerms>
      <aspectTerm term="appetizers" polarity="positive" from="8" to="18"/>
      <aspectTerm term="salads" polarity="conflict" from="23" to="29"/>
    </aspectTerms>
    <aspectCategories>
      <aspectCategory category="food" polarity="positive"/>
    </aspectCategories>
  </sentence>
  <sentence id="814">
    <text>We arrived at nine.</text>
  </sentence>
</sentences>
"""

SEMEVAL_2016 = """<?xml version="1.0" encoding="UTF-8"?>
<Reviews>
  <Review rid="1004293">
    <sentences>
      <sentence id="1004293:0">
        <text>Judging from previous posts this used to be a good place, but not any longer.</text>
        <Opinions>
          <Opinion target="place" category="RESTAURANT#GENERAL" polarity="negative" from="51" to="56"/>
        </Opinions>
      </sentence>
      <sentence id="1004293:1">
        <text>The service was meh.</text>
        <Opinions>
          <Opinion target="service" category="SERVICE#GENERAL" polarity="neutral" from="4" to="11"/>
          <Opinion target="NULL" category="FOOD#QUALITY" polarity="negative" from="0" to="0"/>
        </Opinions>
      </sentence>
      <sentence id="1004293:2">
        <text>It closed at noon.</text>
      </sentence>
    </sentences>
  </Review>
</Reviews>
"""

# Format documented in the ASTE-Data-V2 repository (Xu et al. 2020).
ASTE = (
    "The screen is very large and crystal clear with amazing colors and resolution ."
    "####[([1], [4], 'POS'), ([1], [7], 'POS'), ([10], [9], 'POS'), ([12], [9], 'POS')]\n"
    "Great battery life but poor screen quality ."
    "####[([1, 2], [0], 'POS'), ([5, 6], [4], 'NEG')]\n"
)

# Lines taken verbatim from the official Restaurant-ACOS dev split.
ACOS = (
    "ca n ' t wait wait for my next visit .\t-1,-1 RESTAURANT#GENERAL 2 -1,-1\n"
    "the spicy tuna roll was unusually good and the rock shrimp tempura was awesome ,"
    " great appetizer to share !\t1,4 FOOD#QUALITY 2 6,7\t9,12 FOOD#QUALITY 2 13,14\n"
)


class TestJsonl:
    def test_roundtrip(self, tmp_path):
        examples = [
            ABSAExample(
                text="The pasta was great",
                tuples=[
                    SentimentTuple(
                        aspect=Span("pasta", 4, 9),
                        category="FOOD#QUALITY",
                        opinion=Span("great", 14, 19),
                        polarity="positive",
                    ),
                    SentimentTuple(aspect=IMPLICIT, opinion=IMPLICIT, polarity="negative"),
                ],
                id="x1",
                meta={"lang": "en"},
            ),
            ABSAExample(text="No opinions here."),
        ]
        path = tmp_path / "data.jsonl"
        write_jsonl(examples, path)
        assert read_jsonl(path) == examples

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"text": "a", "tuples": []}\n\n{"text": "b", "tuples": []}\n')
        assert [e.text for e in read_jsonl(path)] == ["a", "b"]

    def test_bad_line_reports_position(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"text": "ok", "tuples": []}\nnot json\n')
        with pytest.raises(DataFormatError, match=":2"):
            read_jsonl(path)


class TestSemEval2014:
    @pytest.fixture()
    def xml_path(self, tmp_path):
        path = tmp_path / "rest14.xml"
        path.write_text(SEMEVAL_2014)
        return path

    def test_terms_view(self, xml_path):
        examples = read_semeval_2014(xml_path)
        assert len(examples) == 2
        first = examples[0]
        assert first.id == "813"
        assert len(first.tuples) == 2
        t = first.tuples[0]
        assert t.aspect == Span("appetizers", 8, 18)
        assert t.polarity == "positive"
        assert t.category is None
        assert first.tuples[1].polarity == "conflict"
        # offset-sliced surface text matches the attribute
        assert first.text[8:18] == "appetizers"
        assert examples[1].tuples == []

    def test_categories_view(self, xml_path):
        examples = read_semeval_2014(xml_path, annotations="categories")
        t = examples[0].tuples[0]
        assert is_implicit(t.aspect)
        assert t.category == "food"
        assert t.polarity == "positive"

    def test_both_views_stay_unlinked(self, xml_path):
        examples = read_semeval_2014(xml_path, annotations="both")
        tuples = examples[0].tuples
        assert len(tuples) == 3
        # no tuple carries both a span aspect and a category
        assert all(t.category is None or is_implicit(t.aspect) for t in tuples)

    def test_invalid_annotations_argument(self, xml_path):
        with pytest.raises(ValueError):
            read_semeval_2014(xml_path, annotations="everything")

    def test_malformed_xml(self, tmp_path):
        path = tmp_path / "broken.xml"
        path.write_text("<sentences><sentence>")
        with pytest.raises(DataFormatError):
            read_semeval_2014(path)


class TestSemEval2016:
    @pytest.fixture()
    def xml_path(self, tmp_path):
        path = tmp_path / "rest16.xml"
        path.write_text(SEMEVAL_2016)
        return path

    def test_opinions(self, xml_path):
        examples = read_semeval_2015(xml_path)
        assert len(examples) == 3
        first = examples[0]
        assert first.meta["review_id"] == "1004293"
        t = first.tuples[0]
        assert t.aspect == Span("place", 51, 56)
        assert t.category == "RESTAURANT#GENERAL"
        assert t.polarity == "negative"

    def test_null_target_is_implicit(self, xml_path):
        examples = read_semeval_2015(xml_path)
        implicit = examples[1].tuples[1]
        assert is_implicit(implicit.aspect)
        assert implicit.category == "FOOD#QUALITY"

    def test_sentence_without_opinions_kept(self, xml_path):
        examples = read_semeval_2015(xml_path)
        assert examples[2].tuples == []


class TestAste:
    @pytest.fixture()
    def aste_path(self, tmp_path):
        path = tmp_path / "train_triplets.txt"
        path.write_text(ASTE)
        return path

    def test_single_token_spans(self, aste_path):
        examples = read_aste(aste_path)
        first = examples[0]
        assert len(first.tuples) == 4
        t = first.tuples[0]
        assert t.aspect.text == "screen"
        assert t.opinion.text == "large"
        assert t.polarity == "positive"
        # offsets really point into the text
        assert first.text[t.aspect.start : t.aspect.end] == "screen"

    def test_multi_token_spans(self, aste_path):
        examples = read_aste(aste_path)
        second = examples[1]
        assert second.tuples[0].aspect.text == "battery life"
        assert second.tuples[0].opinion.text == "Great"
        assert second.tuples[1].aspect.text == "screen quality"
        assert second.tuples[1].polarity == "negative"

    def test_missing_separator(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("no separator here\n")
        with pytest.raises(DataFormatError, match="separator"):
            read_aste(path)

    def test_out_of_bounds_index(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("short sentence .####[([9], [0], 'POS')]\n")
        with pytest.raises(DataFormatError, match="out of bounds"):
            read_aste(path)

    def test_unsafe_payload_rejected(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("a b .####__import__('os')\n")
        with pytest.raises(DataFormatError):
            read_aste(path)


class TestAcos:
    @pytest.fixture()
    def acos_path(self, tmp_path):
        path = tmp_path / "rest16_quad_dev.tsv"
        path.write_text(ACOS)
        return path

    def test_implicit_quadruple(self, acos_path):
        examples = read_acos(acos_path)
        t = examples[0].tuples[0]
        assert is_implicit(t.aspect)
        assert is_implicit(t.opinion)
        assert t.category == "RESTAURANT#GENERAL"
        assert t.polarity == "positive"

    def test_explicit_quadruples(self, acos_path):
        examples = read_acos(acos_path)
        tuples = examples[1].tuples
        assert len(tuples) == 2
        assert tuples[0].aspect.text == "spicy tuna roll"
        assert tuples[0].opinion.text == "good"
        assert tuples[1].aspect.text == "rock shrimp tempura"
        assert tuples[1].opinion.text == "awesome"
        text = examples[1].text
        a = tuples[1].aspect
        assert text[a.start : a.end] == "rock shrimp tempura"

    def test_sentiment_codes(self, tmp_path):
        path = tmp_path / "codes.tsv"
        path.write_text(
            "food was ok .\t0,1 FOOD#QUALITY 1 2,4\nfood was bad .\t0,1 FOOD#QUALITY 0 2,3\n"
        )
        examples = read_acos(path)
        assert examples[0].tuples[0].polarity == "neutral"
        assert examples[1].tuples[0].polarity == "negative"

    def test_line_without_quads(self, tmp_path):
        path = tmp_path / "bad.tsv"
        path.write_text("just a sentence\n")
        with pytest.raises(DataFormatError, match="no quadruples"):
            read_acos(path)

    def test_malformed_quad(self, tmp_path):
        path = tmp_path / "bad.tsv"
        path.write_text("a b c\t0,1 FOOD#QUALITY 2\n")
        with pytest.raises(DataFormatError, match="4 space-separated"):
            read_acos(path)


class TestLoadExamples:
    def test_dispatch(self, tmp_path):
        path = tmp_path / "rest16.xml"
        path.write_text(SEMEVAL_2016)
        for name in ("semeval2016", "SemEval-2016", "semeval_2015"):
            assert len(load_examples(path, name)) == 3

    def test_kwargs_passthrough(self, tmp_path):
        path = tmp_path / "rest14.xml"
        path.write_text(SEMEVAL_2014)
        examples = load_examples(path, "semeval2014", annotations="categories")
        assert examples[0].tuples[0].category == "food"

    def test_unknown_format(self, tmp_path):
        with pytest.raises(ValueError, match="unknown format"):
            load_examples(tmp_path / "x", "parquet")
