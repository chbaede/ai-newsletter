from ai_newsletter.models import Article, NewsletterIssue
from ai_newsletter.priority import assess_priority, priority_summary


def test_assess_priority_levels():
    a_crit = Article(
        title="OpenAI announces GPT-5 release",
        url="https://openai.com",
        source="OpenAI",
        priority_score=88.0,
    )
    p_crit = assess_priority(a_crit)
    assert p_crit.level == "critical"
    assert p_crit.label_ko == "최우선"
    assert p_crit.label_en == "Critical"

    a_high = Article(
        title="Mistral releases new model",
        url="https://mistral.ai",
        source="Mistral",
        priority_score=68.0,
    )
    p_high = assess_priority(a_high)
    assert p_high.level == "high"
    assert p_high.label_ko == "높음"
    assert p_high.label_en == "High"

    a_med = Article(
        title="General AI discussion",
        url="https://example.com",
        source="Blog",
        priority_score=50.0,
    )
    p_med = assess_priority(a_med)
    assert p_med.level == "medium"

    a_watch = Article(
        title="Minor blog update",
        url="https://example.com/2",
        source="Blog",
        priority_score=35.0,
    )
    p_watch = assess_priority(a_watch)
    assert p_watch.level == "watch"


def test_priority_summary():
    issue = NewsletterIssue(
        issue_date="2026-09-17",
        articles=[
            Article(title="Art 1", url="u1", source="s1", priority_score=85.0),
            Article(title="Art 2", url="u2", source="s2", priority_score=65.0),
            Article(title="Art 3", url="u3", source="s3", priority_score=48.0),
        ],
    )
    summary = priority_summary(issue)
    assert summary.counts["critical"] == 1
    assert summary.counts["high"] == 1
    assert summary.counts["medium"] == 1
    assert len(summary.top_items) == 3

