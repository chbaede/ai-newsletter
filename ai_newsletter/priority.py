from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import Article, NewsletterIssue
from .scoring import build_score_explanation, compute_multi_dimensional_scores


@dataclass(frozen=True, slots=True)
class PrioritySignal:
    level: str
    label_ko: str
    label_en: str
    score: int
    reason_ko: str
    reason_en: str
    bullet_reasons_ko: list[str] = field(default_factory=list)
    bullet_reasons_en: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PrioritizedArticle:
    article: Article
    priority: PrioritySignal


@dataclass(frozen=True, slots=True)
class PrioritySummary:
    counts: dict[str, int]
    top_items: list[PrioritizedArticle]


LEVELS = {
    "critical": ("최우선", "Critical", 4),
    "high": ("높음", "High", 3),
    "medium": ("보통", "Medium", 2),
    "watch": ("관찰", "Watch", 1),
}


def assess_priority(
    article: Article,
    past_articles: list[Article] | None = None,
    ref_time: datetime | None = None,
    half_life_hours: float = 36.0,
) -> PrioritySignal:
    """Convert priority_score from scoring.py into presentation-friendly priority level and rationale."""
    if article.priority_score == 0.0 and article.relevance_score == 0.0 and article.impact_score == 0.0:
        scores = compute_multi_dimensional_scores(
            article,
            past_articles=past_articles,
            ref_time=ref_time,
            half_life_hours=half_life_hours,
        )
        article.source_score = scores.source_score
        article.relevance_score = scores.relevance_score
        article.impact_score = scores.impact_score
        article.novelty_score = scores.novelty_score
        article.recency_score = scores.recency_score
        article.priority_score = scores.priority_score
        article.score = scores.priority_score
        explanation = scores.explanation
    else:
        explanation = build_score_explanation(
            source_score=article.source_score,
            relevance_score=article.relevance_score,
            impact_score=article.impact_score,
            novelty_score=article.novelty_score,
            recency_score=article.recency_score,
            priority_score=article.priority_score,
        )

    priority_score = article.priority_score

    if priority_score >= 80.0:
        level = "critical"
    elif priority_score >= 60.0:
        level = "high"
    elif priority_score >= 45.0:
        level = "medium"
    else:
        level = "watch"

    label_ko = LEVELS[level][0]
    label_en = LEVELS[level][1]

    return PrioritySignal(
        level=level,
        label_ko=label_ko,
        label_en=label_en,
        score=int(round(priority_score)),
        reason_ko=explanation.summary_ko,
        reason_en=explanation.summary_en,
        bullet_reasons_ko=explanation.reasons_ko,
        bullet_reasons_en=explanation.reasons_en,
    )


def priority_summary(issue: NewsletterIssue) -> PrioritySummary:
    prioritized = [
        PrioritizedArticle(article=article, priority=assess_priority(article))
        for article in issue.articles
    ]
    counts = {level: 0 for level in LEVELS}
    for item in prioritized:
        counts[item.priority.level] += 1
    prioritized.sort(
        key=lambda item: (
            LEVELS[item.priority.level][2],
            item.priority.score,
            item.article.score,
        ),
        reverse=True,
    )
    return PrioritySummary(counts=counts, top_items=prioritized[:5])
