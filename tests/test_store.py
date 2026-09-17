from datetime import datetime, timezone
import pytest

from ai_newsletter.models import Article, Event, EventArticle, NewsletterIssue
from ai_newsletter.store import NewsletterStore


@pytest.fixture
def test_store(tmp_path):
    db_file = tmp_path / "test_newsletter.db"
    return NewsletterStore(db_file)


def test_store_save_and_retrieve_issue(test_store):
    articles = [
        Article(
            title="OpenAI announces GPT-4o flagship model",
            title_ko="오픈AI, 플래그십 모델 GPT-4o 공개",
            title_en="OpenAI Announces GPT-4o Flagship Model",
            url="https://openai.com/index/gpt-4o",
            source="OpenAI",
            category="frontier_models",
            summary_ko="오픈AI가 음성과 비전을 지원하는 GPT-4o를 공개했습니다.",
            summary_en="OpenAI announced GPT-4o with native multimodal features.",
            why_it_matters_ko="멀티모달 인터랙션의 지연시간을 획기적으로 개선했습니다.",
            why_it_matters_en="Dramatically reduces latency for multimodal interactions.",
            priority_score=92.0,
            tags=["OpenAI", "LLM"],
        )
    ]
    events = [
        Event(
            event_id="ev_test1",
            title="OpenAI GPT-4o Launch",
            category="frontier_models",
            importance=92.0,
        )
    ]
    event_articles = [
        EventArticle(event_id="ev_test1", article_id=articles[0].article_id, relationship="primary")
    ]

    issue = NewsletterIssue(
        issue_date="2026-09-17",
        articles=articles,
        events=events,
        warnings=["Test note"],
    )

    test_store.save_issue(issue, event_articles=event_articles)

    retrieved = test_store.get_issue("2026-09-17")
    assert retrieved is not None
    assert retrieved.issue_date == "2026-09-17"
    assert len(retrieved.articles) == 1
    assert retrieved.articles[0].title_ko == "오픈AI, 플래그십 모델 GPT-4o 공개"
    assert retrieved.articles[0].title_en == "OpenAI Announces GPT-4o Flagship Model"
    assert retrieved.articles[0].summary_ko == "오픈AI가 음성과 비전을 지원하는 GPT-4o를 공개했습니다."
    assert retrieved.articles[0].summary_en == "OpenAI announced GPT-4o with native multimodal features."
    assert retrieved.articles[0].why_it_matters_ko == "멀티모달 인터랙션의 지연시간을 획기적으로 개선했습니다."
    assert retrieved.articles[0].why_it_matters_en == "Dramatically reduces latency for multimodal interactions."
    assert len(retrieved.events) == 1
    assert retrieved.events[0].event_id == "ev_test1"


def test_store_mail_settings(test_store):
    test_store.update_mail_settings({"smtp_host": "smtp.gmail.com", "smtp_port": 587})
    settings = test_store.mail_settings()
    assert settings["smtp_host"] == "smtp.gmail.com"
    assert settings["smtp_port"] == "587"


def test_scheduler_runs(test_store):
    assert test_store.has_daily_run_completed("2026-09-17") is False
    acquired = test_store.acquire_daily_run("2026-09-17")
    assert acquired is True

    # Re-acquire while running should fail
    assert test_store.acquire_daily_run("2026-09-17") is False

    test_store.finish_daily_run("2026-09-17", metrics={"articles": 10})
    assert test_store.has_daily_run_completed("2026-09-17") is True

