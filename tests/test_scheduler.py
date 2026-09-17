from datetime import datetime
from zoneinfo import ZoneInfo

from ai_newsletter.scheduler import parse_time_string, should_run_daily_collection
from ai_newsletter.store import NewsletterStore


def test_parse_time_string():
    h, m = parse_time_string("06:00")
    assert h == 6
    assert m == 0


def test_should_run_daily_collection(tmp_path):
    store = NewsletterStore(tmp_path / "test.db")
    tz = ZoneInfo("Europe/Berlin")

    # Before scheduled time (05:00 < 06:00)
    dt_early = datetime(2026, 9, 17, 5, 0, tzinfo=tz)
    should, date_str, reason = should_run_daily_collection(dt_early, "06:00", store)
    assert should is False
    assert reason == "before_scheduled_time"

    # After scheduled time (08:00 >= 06:00), not run yet
    dt_late = datetime(2026, 9, 17, 8, 0, tzinfo=tz)
    should, date_str, reason = should_run_daily_collection(dt_late, "06:00", store)
    assert should is True

    # Mark completed
    store.finish_daily_run("2026-09-17")
    should, date_str, reason = should_run_daily_collection(dt_late, "06:00", store)
    assert should is False
    assert reason == "already_completed"

    # Forced run
    should, date_str, reason = should_run_daily_collection(dt_late, "06:00", store, force=True)
    assert should is True
    assert reason == "forced"


