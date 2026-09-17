from datetime import date
from ai_newsletter.clustering import cluster_articles
from ai_newsletter.collector import collect_from_entries
from ai_newsletter.models import FeedEntry, NewsletterIssue
from ai_newsletter.presentation import build_intelligence_sections
from ai_newsletter.store import NewsletterStore


def test_e2e_pipeline_in_memory(tmp_path):
    store = NewsletterStore(tmp_path / "e2e.db")

    raw_entries = [
        FeedEntry(
            title="OpenAI releases o1 reasoning model with state-of-the-art math capabilities",
            url="https://openai.com/index/learning-to-reason-with-llms",
            source="OpenAI Newsroom",
            bucket="frontier_models",
            source_type="official",
            authority_score=95,
            excerpt="o1 models use reinforcement learning to generate chain of thought before answering.",
        ),
        FeedEntry(
            title="OpenAI launches o1 reasoning model to solve complex math problems",
            url="https://techcrunch.com/openai-o1-math",
            source="TechCrunch",
            bucket="frontier_models",
            source_type="media",
            authority_score=85,
            excerpt="New o1 series challenges conventional LLM scaling with inference compute.",
        ),
        FeedEntry(
            title="NVIDIA and TSMC expand Blackwell production to meet AI demand",
            url="https://reuters.com/nvidia-tsmc-blackwell",
            source="Reuters",
            bucket="hardware_infra",
            source_type="media",
            authority_score=95,
            excerpt="Foundry capacity expands for AI chips.",
        ),
    ]

    # 1. Deduplication and classification
    articles, warnings = collect_from_entries(raw_entries)
    assert len(articles) == 3

    # 2. Clustering
    events, event_articles, clustered_articles = cluster_articles(articles)
    assert len(events) == 2  # OpenAI o1 event + NVIDIA Blackwell event

    # Check official source was chosen as primary
    o1_event = next(e for e in events if "o1" in e.title)
    assert o1_event.has_official_source is True
    assert o1_event.source_count == 2

    # 3. Store saving
    issue = NewsletterIssue(
        issue_date="2026-09-17",
        articles=clustered_articles,
        events=events,
    )
    store.save_issue(issue, event_articles=event_articles)

    # 4. Retrieval and presentation
    saved_issue = store.get_issue("2026-09-17")
    assert saved_issue is not None
    assert len(saved_issue.articles) == 3

    sections_ko = build_intelligence_sections(saved_issue, lang="ko")
    sections_en = build_intelligence_sections(saved_issue, lang="en")

    assert len(sections_ko) > 0
    assert len(sections_en) > 0

    # Ensure bilingual titles and summaries are present
    art = saved_issue.articles[0]
    assert art.title_ko
    assert art.title_en
    assert art.summary_ko
    assert art.summary_en

