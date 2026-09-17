from datetime import datetime, timezone

from ai_newsletter.collector import (
    are_titles_similar,
    canonicalize_url,
    clean_title_for_comparison,
    collect_from_entries,
    extract_title_tokens,
    normalize_title,
)
from ai_newsletter.models import Article, FeedEntry


def test_clean_title():
    t1 = "[단독] OpenAI, 차세대 AI 모델 GPT-5 개발 착수 - AI타임스"
    cleaned = clean_title_for_comparison(t1)
    assert "단독" not in cleaned
    assert "ai타임스" not in cleaned
    assert "openai" in cleaned


def test_are_titles_similar():
    t1 = "OpenAI announces GPT-4o with voice and vision"
    t2 = "OpenAI launches GPT-4o with voice and vision features"
    assert are_titles_similar(t1, t2) is True

    t3 = "NVIDIA reveals Blackwell B200 GPU architecture"
    assert are_titles_similar(t1, t3) is False


def test_canonicalize_url():
    url = "https://techcrunch.com/2026/09/15/openai-gpt4o/?utm_source=twitter&utm_campaign=ai&ref=feed"
    clean = canonicalize_url(url)
    assert "utm_source" not in clean
    assert "utm_campaign" not in clean
    assert "ref" not in clean
    assert clean == "https://techcrunch.com/2026/09/15/openai-gpt4o"


def test_collect_from_entries_deduplication():
    entries = [
        FeedEntry(
            title="OpenAI launches GPT-4o multimodal model",
            url="https://techcrunch.com/openai-gpt4o?utm_source=rss",
            source="TechCrunch",
            bucket="frontier_models",
        ),
        FeedEntry(
            title="OpenAI launches GPT-4o multimodal model",
            url="https://techcrunch.com/openai-gpt4o?utm_source=twitter",
            source="TechCrunch",
            bucket="frontier_models",
        ),
        FeedEntry(
            title="NVIDIA announces new Blackwell AI chips",
            url="https://theverge.com/nvidia-blackwell",
            source="The Verge",
            bucket="hardware_infra",
        ),
    ]

    articles, warnings = collect_from_entries(entries)
    assert len(articles) == 2, f"Expected 2 articles after dedupe, got {len(articles)}"

