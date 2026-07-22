# tests/core/test_cleaners.py
"""
Тесты для пайплайна очистки текста.
Покрывают RegexCleaner и TextCleaningPipeline — без моков,
чистая unit-логика без внешних зависимостей.
"""

import pytest

from src.core.data.cleaners import RegexCleaner, TextCleaningPipeline


# ---------------------------------------------------------------------------
# RegexCleaner
# ---------------------------------------------------------------------------


class TestRegexCleaner:
    def test_removes_html_tags(self):
        cleaner = RegexCleaner(pattern="<.*?>", replacement=" ")
        assert cleaner.clean("<b>Hello</b> world") == " Hello  world"

    def test_removes_non_printable_characters(self):
        cleaner = RegexCleaner(pattern="[^\x20-\x7e\n\t]", replacement="")
        text_with_garbage = "Hello\x00World\x1fClean"
        assert cleaner.clean(text_with_garbage) == "HelloWorldClean"

    def test_collapses_whitespace(self):
        cleaner = RegexCleaner(pattern=r"\s+", replacement=" ")
        assert cleaner.clean("too   many    spaces") == "too many spaces"

    def test_empty_string_returns_empty(self):
        cleaner = RegexCleaner(pattern="<.*?>", replacement=" ")
        assert cleaner.clean("") == ""

    def test_no_match_returns_original(self):
        cleaner = RegexCleaner(pattern="<.*?>", replacement=" ")
        original = "plain text without html"
        assert cleaner.clean(original) == original

    def test_replacement_is_applied(self):
        cleaner = RegexCleaner(pattern=r"\d+", replacement="NUM")
        assert cleaner.clean("order 123 placed") == "order NUM placed"

    def test_unicode_text_preserved(self):
        """Символы кириллицы не должны трогаться HTML-клинером."""
        cleaner = RegexCleaner(pattern="<.*?>", replacement="")
        text = "Привет <b>мир</b>"
        assert cleaner.clean(text) == "Привет мир"


# ---------------------------------------------------------------------------
# TextCleaningPipeline
# ---------------------------------------------------------------------------


class TestTextCleaningPipeline:
    def _make_pipeline(self) -> TextCleaningPipeline:
        """Репликация продакшн-конфига из configs/data/default.yaml."""
        return TextCleaningPipeline(
            cleaners=[
                RegexCleaner(pattern="<.*?>", replacement=" "),
                RegexCleaner(pattern="[^\x20-\x7e\n\t]", replacement=""),
                RegexCleaner(pattern=r"\s+", replacement=" "),
            ]
        )

    def test_pipeline_applies_all_steps_in_order(self):
        pipeline = self._make_pipeline()
        raw = "  <b>Hello</b>\x00  world  "
        result = pipeline(raw)
        assert result == " Hello world "

    def test_pipeline_with_empty_cleaners_is_identity(self):
        pipeline = TextCleaningPipeline(cleaners=[])
        text = "unchanged"
        assert pipeline(text) == text

    def test_pipeline_html_then_whitespace(self):
        """Порядок важен: сначала strip HTML, потом collapse spaces."""
        pipeline = self._make_pipeline()
        raw = "<p>  Breaking   news  </p>"
        result = pipeline(raw)
        # После HTML-strip → " Breaking   news  ", после collapse → " Breaking news "
        assert "Breaking news" in result
        assert "<p>" not in result

    def test_pipeline_is_callable(self):
        pipeline = self._make_pipeline()
        assert callable(pipeline)

    def test_pipeline_single_cleaner(self):
        pipeline = TextCleaningPipeline(cleaners=[RegexCleaner(pattern=r"\d", replacement="")])
        assert pipeline("abc123def456") == "abcdef"

    @pytest.mark.parametrize(
        "text",
        [
            "FREE MONEY CLICK NOW!!!",
            "Normal email about project update.",
            "",
            "   ",
            "<html><body>Spam</body></html>",
        ],
    )
    def test_pipeline_does_not_raise_on_various_inputs(self, text):
        """Пайплайн не должен падать ни на каком текстовом вводе."""
        pipeline = self._make_pipeline()
        result = pipeline(text)
        assert isinstance(result, str)
