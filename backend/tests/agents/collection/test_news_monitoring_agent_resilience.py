"""NewsMonitoringAgent: analysis flow, graceful degradation, and contract/static guards.

Root cause these tests guard against: the agent used to call `TextProcessor.extract_location`
(which no longer exists) and reference an undefined `analysis` variable, so no real article could
ever be saved. Bare `Mock()` doubles hid that (any attribute "exists"), so several tests here use
`create_autospec` doubles of the REAL TextProcessor / Geocoder, and a static check looks for names
that are used but never defined.
"""
from __future__ import annotations

import builtins
from pathlib import Path
import symtable
from unittest.mock import Mock, create_autospec

import pytest

import src.agents.collection.news_monitoring_agent as agent_module
import src.external.news.news_client as client_module
from src.agents.collection.news_monitoring_agent import NewsMonitoringAgent
from src.external.geocoding.geocoding_client import Geocoder
from src.external.news.news_client import RSSFetcher, TextProcessor
from src.models.fire_report import WildfireReport
from src.models.news_text_analysis import NewsTextAnalysis
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.repositories.exceptions import NewsRepositoryError
from src.repositories.news_repository import NewsRepository, SaveNewsReportResult

STRONG = NewsWildfireSignalStrength.STRONG
WEAK = NewsWildfireSignalStrength.WEAK
UNAVAILABLE = NewsTextAnalysis(location_name=None, wildfire_signal_strength=None)


def entry(n: int = 1, **overrides) -> dict:
    values = {
        "source_feed": "Feed",
        "title": f"שריפה {n}",
        "summary": f"כוחות כיבוי {n}",
        "link": f"https://example.com/news/{n}",
        "published": "Sat, 12 Sep 2026 10:00:00 GMT",
    }
    values.update(overrides)
    return values


class Wiring:
    """Autospec'd doubles of the real collaborators, wired for a successful run."""

    def __init__(self, entries):
        self.rss = create_autospec(RSSFetcher, instance=True)
        self.rss.fetch_all.return_value = list(entries)
        self.text = create_autospec(TextProcessor, instance=True)
        self.text.is_relevant.return_value = True
        self.text.analyze.return_value = NewsTextAnalysis("כרמל", STRONG)
        self.text.translate_report.side_effect = lambda title, summary, location: (title, summary, location)
        self.geocoder = create_autospec(Geocoder, instance=True)
        self.geocoder.geocode.return_value = (32.7, 35.0)
        self.repo = create_autospec(NewsRepository, instance=True)
        self.repo.exists_by_source_url.return_value = False
        self.repo.save_report.side_effect = lambda report: SaveNewsReportResult(report=report, is_duplicate=False)

    def agent(self) -> NewsMonitoringAgent:
        return NewsMonitoringAgent(
            rss_fetcher=self.rss,
            text_processor=self.text,
            geocoder=self.geocoder,
            news_repository=self.repo,
            enable_file_logging=False,
        )

    def saved(self) -> list[WildfireReport]:
        return [call.args[0] for call in self.repo.save_report.call_args_list]


# --- the analysis flow ---


def test_an_analysed_article_is_saved_with_its_location_and_signal_strength():
    wiring = Wiring([entry(1)])

    saved_count = wiring.agent().run_once()

    assert saved_count == 1
    (report,) = wiring.saved()
    assert report.wildfire_signal_strength is STRONG
    assert report.location_name == "כרמל"
    assert (report.latitude, report.longitude) == (32.7, 35.0)


def test_analyze_is_called_exactly_once_per_new_article():
    wiring = Wiring([entry(1), entry(2), entry(3)])

    wiring.agent().run_once()

    assert wiring.text.analyze.call_count == 3
    assert [call.args for call in wiring.text.analyze.call_args_list] == [
        ("שריפה 1", "כוחות כיבוי 1"),
        ("שריפה 2", "כוחות כיבוי 2"),
        ("שריפה 3", "כוחות כיבוי 3"),
    ]
    assert wiring.text.translate_report.call_count == 3


def test_each_article_gets_its_own_analysis_with_no_state_leaking_between_articles():
    wiring = Wiring([entry(1), entry(2), entry(3)])
    wiring.text.analyze.side_effect = [
        NewsTextAnalysis("חיפה", WEAK),
        UNAVAILABLE,
        NewsTextAnalysis("כרמל", STRONG),
    ]

    wiring.agent().run_once()

    first, second, third = wiring.saved()
    assert (first.location_name, first.wildfire_signal_strength) == ("חיפה", WEAK)
    assert (second.location_name, second.wildfire_signal_strength) == (None, None)
    assert (third.location_name, third.wildfire_signal_strength) == ("כרמל", STRONG)


def test_geocoding_and_translation_use_the_analysed_hebrew_location():
    wiring = Wiring([entry(1)])

    wiring.agent().run_once()

    wiring.geocoder.geocode.assert_called_once_with("כרמל")
    wiring.text.translate_report.assert_called_once_with("שריפה 1", "כוחות כיבוי 1", "כרמל")


def test_the_agent_only_uses_methods_that_exist_on_the_real_collaborators():
    """create_autospec refuses unknown attributes: a call to a removed method (like extract_location) fails."""
    wiring = Wiring([entry(1)])
    assert not hasattr(TextProcessor, "extract_location")

    with pytest.raises(AttributeError):
        wiring.text.extract_location("t", "s")
    assert wiring.agent().run_once() == 1


# --- irrelevant and duplicate articles ---


def test_irrelevant_articles_are_skipped_before_any_external_call():
    wiring = Wiring([entry(1), entry(2)])
    wiring.text.is_relevant.side_effect = [False, True]

    saved_count = wiring.agent().run_once()

    assert saved_count == 1
    assert wiring.text.analyze.call_count == 1
    assert [report.source_url for report in wiring.saved()] == ["https://example.com/news/2"]


def test_an_already_known_url_is_skipped_before_analysis_geocoding_and_translation():
    wiring = Wiring([entry(1), entry(2)])
    wiring.repo.exists_by_source_url.side_effect = lambda url: url.endswith("/1")

    saved_count = wiring.agent().run_once()

    assert saved_count == 1
    assert wiring.text.analyze.call_count == 1
    assert wiring.geocoder.geocode.call_count == 1
    assert wiring.text.translate_report.call_count == 1
    assert [report.source_url for report in wiring.saved()] == ["https://example.com/news/2"]


def test_a_duplicate_reported_by_the_repository_is_not_counted():
    wiring = Wiring([entry(1), entry(1)])  # the same link twice in one feed batch
    wiring.repo.save_report.side_effect = [
        SaveNewsReportResult(report=Mock(spec=WildfireReport), is_duplicate=False),
        SaveNewsReportResult(report=Mock(spec=WildfireReport), is_duplicate=True),
    ]

    assert wiring.agent().run_once() == 1


def test_an_entry_without_a_link_is_skipped_without_processing():
    wiring = Wiring([entry(1, link=""), entry(2)])

    assert wiring.agent().run_once() == 1
    assert wiring.text.analyze.call_count == 1


# --- failed analysis / geocoding: handled the way the existing design says ---


def test_unavailable_analysis_still_saves_the_report_with_an_unknown_signal():
    wiring = Wiring([entry(1)])
    wiring.text.analyze.return_value = UNAVAILABLE
    wiring.geocoder.geocode.return_value = (None, None)

    assert wiring.agent().run_once() == 1
    (report,) = wiring.saved()
    assert report.wildfire_signal_strength is None  # unknown - never fabricated as NONE
    assert report.location_name is None and report.latitude is None and report.longitude is None
    wiring.geocoder.geocode.assert_called_once_with(None)


def test_failed_geocoding_still_saves_the_report_with_null_coordinates_and_keeps_the_signal():
    wiring = Wiring([entry(1)])
    wiring.geocoder.geocode.return_value = (None, None)  # Geocoder's documented "miss or failure" result

    assert wiring.agent().run_once() == 1
    (report,) = wiring.saved()
    assert report.location_name == "כרמל"
    assert (report.latitude, report.longitude) == (None, None)
    assert report.wildfire_signal_strength is STRONG


# --- one bad article must not stop the rest ---


def test_an_unexpected_analysis_crash_skips_only_that_article():
    wiring = Wiring([entry(1), entry(2), entry(3)])
    wiring.text.analyze.side_effect = [
        NewsTextAnalysis("חיפה", WEAK),
        RuntimeError("LLM client blew up"),
        NewsTextAnalysis("כרמל", STRONG),
    ]

    saved_count = wiring.agent().run_once()

    assert saved_count == 2
    assert [report.source_url for report in wiring.saved()] == ["https://example.com/news/1", "https://example.com/news/3"]


def test_an_unexpected_geocoder_crash_skips_only_that_article():
    wiring = Wiring([entry(1), entry(2)])
    wiring.geocoder.geocode.side_effect = [RuntimeError("geocoder blew up"), (32.7, 35.0)]

    assert wiring.agent().run_once() == 1
    assert [report.source_url for report in wiring.saved()] == ["https://example.com/news/2"]


def test_an_unexpected_translation_crash_skips_only_that_article():
    wiring = Wiring([entry(1), entry(2)])
    wiring.text.translate_report.side_effect = [
        RuntimeError("translator blew up"),
        ("Title 2", "Summary 2", "Carmel"),
    ]

    assert wiring.agent().run_once() == 1
    assert wiring.saved()[0].title == "Title 2"


def test_a_malformed_entry_missing_fields_is_skipped_and_the_cycle_continues():
    broken_no_title = {"source_feed": "Feed", "link": "https://example.com/x", "summary": "s", "published": ""}
    broken_no_link = {"source_feed": "Feed", "title": "שריפה", "summary": "s", "published": ""}
    wiring = Wiring([broken_no_title, entry(2), broken_no_link, entry(3)])

    saved_count = wiring.agent().run_once()

    assert saved_count == 2
    assert [report.source_url for report in wiring.saved()] == ["https://example.com/news/2", "https://example.com/news/3"]


def test_a_non_dict_entry_is_skipped():
    wiring = Wiring([None, "garbage", entry(1)])

    assert wiring.agent().run_once() == 1


def test_a_failing_relevance_check_treats_that_entry_as_irrelevant():
    wiring = Wiring([entry(1), entry(2)])
    wiring.text.is_relevant.side_effect = [RuntimeError("bad text"), True]

    assert wiring.agent().run_once() == 1
    assert [report.source_url for report in wiring.saved()] == ["https://example.com/news/2"]


def test_a_repository_failure_on_the_existence_check_skips_only_that_article():
    wiring = Wiring([entry(1), entry(2)])
    wiring.repo.exists_by_source_url.side_effect = [NewsRepositoryError("db down"), False]

    assert wiring.agent().run_once() == 1
    assert [report.source_url for report in wiring.saved()] == ["https://example.com/news/2"]


def test_a_repository_failure_on_save_skips_only_that_article():
    wiring = Wiring([entry(1), entry(2)])
    wiring.repo.save_report.side_effect = [
        NewsRepositoryError("write failed"),
        SaveNewsReportResult(report=Mock(spec=WildfireReport), is_duplicate=False),
    ]

    assert wiring.agent().run_once() == 1
    assert wiring.repo.save_report.call_count == 2


def test_an_analysis_object_that_cannot_build_a_valid_report_skips_only_that_article():
    wiring = Wiring([entry(1), entry(2)])
    wiring.text.analyze.side_effect = [
        NewsTextAnalysis("חיפה", WEAK),
        Mock(location_name="כרמל", wildfire_signal_strength="not-an-enum"),  # WildfireReport refuses this
    ]

    assert wiring.agent().run_once() == 1


def test_an_empty_or_fully_broken_cycle_does_not_raise():
    assert Wiring([]).agent().run_once() == 0

    wiring = Wiring([entry(1), entry(2)])
    wiring.text.analyze.side_effect = RuntimeError("everything is down")
    assert wiring.agent().run_once() == 0


# --- static guard: no name may be used without being defined ---


def undefined_global_names(source: str, defined: set[str]) -> set[tuple[str, str]]:
    """(scope, name) for every name that is read from the global scope but never defined."""
    available = defined | set(dir(builtins))
    missing: set[tuple[str, str]] = set()

    def visit(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in available:
                missing.add((table.get_name(), symbol.get_name()))
        for child in table.get_children():
            visit(child)

    visit(symtable.symtable(source, "<module>", "exec"))
    return missing


def test_the_static_check_catches_a_name_used_before_it_is_assigned():
    buggy = (
        "def run_once(items):\n"
        "    for item in items:\n"
        "        location = item.extract_location()\n"
        "        report = build(signal=analysis.strength)\n"  # `analysis` is never assigned
        "    return report\n"
    )

    assert ("run_once", "analysis") in undefined_global_names(buggy, {"build"})


@pytest.mark.parametrize("module", [agent_module, client_module], ids=["news_monitoring_agent", "news_client"])
def test_the_news_modules_use_no_undefined_names(module):
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert undefined_global_names(source, set(vars(module))) == set()


def test_the_analysis_is_assigned_before_it_is_used_in_the_agent():
    import ast

    tree = ast.parse(Path(agent_module.__file__).read_text(encoding="utf-8"))
    process = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_process_entry")
    events = []
    for node in ast.walk(process):
        if isinstance(node, ast.Name) and node.id == "analysis":
            events.append((node.lineno, node.col_offset, type(node.ctx).__name__))
    events.sort()

    assert events, "the agent no longer refers to `analysis`"
    assert events[0][2] == "Store"  # the first mention is the assignment
    assert sum(1 for _, _, kind in events if kind == "Store") == 1  # assigned exactly once (one LLM call)
