"""Tests for custom-data interop: records, CSV, JSON, pandas, HF datasets."""

import json

import pytest

from aspectkit.exceptions import DataFormatError
from aspectkit.io import (
    from_hf_dataset,
    from_pandas,
    from_records,
    read_csv,
    read_json,
    to_pandas,
    to_records,
)
from aspectkit.schema import IMPLICIT, ABSAExample, SentimentTuple, Span, is_implicit

NESTED = [
    {
        "id": "r1",
        "text": "The pasta was great",
        "tuples": [
            {
                "aspect": "pasta",
                "category": "FOOD#QUALITY",
                "opinion": "great",
                "polarity": "positive",
            },
            {"aspect": None, "polarity": "negative"},
        ],
    },
    {"id": "r2", "text": "We arrived at nine.", "tuples": []},
]

FLAT = [
    {"text": "Good pasta, bad wine", "aspect": "pasta", "polarity": "positive"},
    {"text": "Good pasta, bad wine", "aspect": "wine", "polarity": "negative"},
    {"text": "Service was fine", "aspect": "Service", "polarity": "neutral"},
]


class TestNestedRecords:
    def test_basic_conversion(self):
        examples = from_records(NESTED)
        assert len(examples) == 2
        first = examples[0]
        assert first.id == "r1"
        explicit, implicit = first.tuples
        assert explicit.aspect.text == "pasta"
        assert explicit.aspect.aligned  # string spans get offset-aligned
        assert explicit.category == "FOOD#QUALITY"
        assert is_implicit(implicit.aspect)
        assert examples[1].tuples == []

    def test_span_dicts_taken_verbatim(self):
        examples = from_records(
            [
                {
                    "text": "ab pasta",
                    "tuples": [
                        {"aspect": {"text": "pasta", "start": 3, "end": 8}, "polarity": "positive"}
                    ],
                }
            ]
        )
        assert examples[0].tuples[0].aspect == Span("pasta", 3, 8)

    def test_custom_field_names(self):
        records = [
            {
                "sentence": "Great screen",
                "labels": [{"target": "screen", "sentiment": "POS"}],
            }
        ]
        examples = from_records(
            records, text="sentence", tuples="labels", aspect="target", polarity="sentiment"
        )
        t = examples[0].tuples[0]
        assert t.aspect.text == "screen"
        assert t.polarity == "positive"

    def test_tuples_field_must_be_list(self):
        with pytest.raises(DataFormatError, match="list of opinion dicts"):
            from_records([{"text": "x", "tuples": "pasta"}])

    def test_opinion_entries_must_be_dicts(self):
        with pytest.raises(DataFormatError, match="dicts"):
            from_records([{"text": "x", "tuples": ["pasta"]}])


class TestFlatRecords:
    def test_grouped_by_text(self):
        examples = from_records(FLAT)
        assert len(examples) == 2
        assert [t.aspect.text for t in examples[0].tuples] == ["pasta", "wine"]
        assert examples[1].tuples[0].polarity == "neutral"

    def test_grouped_by_id_when_present(self):
        records = [
            {"id": "a", "text": "same text", "aspect": "x", "polarity": "positive"},
            {"id": "b", "text": "same text", "aspect": "y", "polarity": "negative"},
        ]
        examples = from_records(records)
        assert len(examples) == 2  # distinct ids beat identical text
        assert examples[0].id == "a"

    def test_group_by_field(self):
        records = [
            {"rev": 1, "text": "t1", "aspect": "a", "polarity": "positive"},
            {"rev": 1, "text": "t1", "aspect": "b", "polarity": "negative"},
        ]
        examples = from_records(records, group_by="rev")
        assert len(examples) == 1
        assert len(examples[0].tuples) == 2

    def test_inconsistent_text_in_group_rejected(self):
        records = [
            {"rev": 1, "text": "t1", "aspect": "a", "polarity": "positive"},
            {"rev": 1, "text": "t2", "aspect": "b", "polarity": "negative"},
        ]
        with pytest.raises(DataFormatError, match="mixes different texts"):
            from_records(records, group_by="rev")

    def test_null_like_aspect_becomes_implicit(self):
        for null in (None, "", "NULL", "implicit", float("nan")):
            examples = from_records([{"text": "x", "aspect": null, "polarity": "positive"}])
            assert is_implicit(examples[0].tuples[0].aspect), repr(null)

    def test_missing_opinion_is_absent_explicit_null_is_implicit(self):
        examples = from_records(
            [
                {"text": "x", "aspect": "a", "polarity": "positive", "opinion": ""},
                {"text": "y", "aspect": "a", "polarity": "positive", "opinion": "NULL"},
            ]
        )
        assert examples[0].tuples[0].opinion is None
        assert is_implicit(examples[1].tuples[0].opinion)

    def test_text_only_record_yields_no_tuple(self):
        examples = from_records([{"text": "nothing here"}], tuples=None)
        assert examples[0].tuples == []

    def test_polarity_labels_normalised(self):
        examples = from_records([{"text": "x", "aspect": "a", "polarity": "NEG"}])
        assert examples[0].tuples[0].polarity == "negative"

    def test_bad_polarity_rejected(self):
        with pytest.raises(DataFormatError, match="cannot map"):
            from_records([{"text": "x", "aspect": "a", "polarity": "great"}])

    def test_missing_text_rejected(self):
        with pytest.raises(DataFormatError, match="missing text"):
            from_records([{"aspect": "a", "polarity": "positive"}])

    def test_explicit_flat_overrides_autodetect(self):
        # records have a "tuples" column that is NOT nested data
        records = [{"text": "x", "tuples": "ignore", "aspect": "a", "polarity": "positive"}]
        examples = from_records(records, tuples=None)
        assert examples[0].tuples[0].aspect.text == "a"


class TestFileEntryPoints:
    def test_read_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text(
            "text,aspect,polarity\n"
            '"Good pasta, bad wine",pasta,positive\n'
            '"Good pasta, bad wine",wine,negative\n'
        )
        examples = read_csv(path)
        assert len(examples) == 1
        assert [t.polarity for t in examples[0].tuples] == ["positive", "negative"]

    def test_read_csv_with_mapping(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("review,term,label\nNice room,room,POS\n")
        examples = read_csv(path, text="review", aspect="term", polarity="label")
        assert examples[0].tuples[0].aspect.text == "room"

    def test_read_json_nested(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps(NESTED))
        examples = read_json(path)
        assert examples[0].id == "r1"
        assert len(examples[0].tuples) == 2

    def test_read_json_requires_array(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"text": "x"}')
        with pytest.raises(DataFormatError, match="JSON array"):
            read_json(path)

    def test_native_export_roundtrip(self, tmp_path):
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
                id="r1",
            )
        ]
        path = tmp_path / "export.json"
        path.write_text(json.dumps(to_records(examples)))
        assert read_json(path) == examples


class TestFrameworkInterop:
    def test_from_pandas(self):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame(FLAT)
        examples = from_pandas(frame)
        assert len(examples) == 2
        assert examples[0].tuples[0].aspect.text == "pasta"

    def test_from_pandas_nan_handling(self):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame(
            [
                {"text": "x", "aspect": "a", "polarity": "positive"},
                {"text": "y", "aspect": None, "polarity": "negative"},
            ]
        )
        examples = from_pandas(frame)  # the None becomes NaN inside pandas
        assert is_implicit(examples[1].tuples[0].aspect)

    def test_from_pandas_rejects_non_frame(self):
        with pytest.raises(TypeError, match="DataFrame"):
            from_pandas([{"text": "x"}])

    def test_from_hf_dataset_duck_typed(self):
        # anything iterating as dict rows works, datasets.Dataset included
        examples = from_hf_dataset(iter(NESTED))
        assert len(examples) == 2

    def test_to_records_flat(self):
        examples = from_records(NESTED)
        rows = to_records(examples, flat=True)
        assert rows[0] == {
            "id": "r1",
            "text": "The pasta was great",
            "aspect": "pasta",
            "category": "FOOD#QUALITY",
            "opinion": "great",
            "polarity": "positive",
        }
        assert rows[1]["aspect"] is None  # implicit
        # example without opinions still produces an accounting row
        assert rows[2]["text"] == "We arrived at nine."
        assert rows[2]["polarity"] is None

    def test_to_pandas(self):
        pytest.importorskip("pandas")
        frame = to_pandas(from_records(NESTED))
        assert list(frame.columns) == ["id", "text", "aspect", "category", "opinion", "polarity"]
        assert len(frame) == 3
