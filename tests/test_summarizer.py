from ai_newsletter.models import Article
from ai_newsletter.summarizer import TemplateSummarizer, classify_article


def test_template_summarizer():
    article = Article(
        title="OpenAI announces GPT-4o flagship model with native multimodal capabilities",
        url="https://openai.com/index/gpt-4o",
        source="OpenAI",
        category="frontier_models",
        excerpt="The new flagship model can reason across audio, vision, and text in real time.",
        tags=["OpenAI", "LLM", "Multimodal"],
        entities=["OpenAI"],
        topics=["LLM", "Multimodal"],
    )

    summarizer = TemplateSummarizer()
    res = summarizer.summarize(article)

    assert res.title_ko
    assert res.title_en
    assert res.summary_ko
    assert res.summary_en
    assert res.why_it_matters_ko
    assert res.why_it_matters_en
    assert len(res.key_points) >= 2


def test_classify_article_end_to_end():
    raw_article = Article(
        title="NVIDIA begins mass shipments of Blackwell B200 accelerators for AI datacenters",
        url="https://techcrunch.com/nvidia-blackwell",
        source="TechCrunch",
        category="hardware_infra",
        excerpt="Cloud hyperscalers have placed massive pre-orders for B200 clusters.",
    )

    processed = classify_article(raw_article)
    assert processed.primary_category == "hardware_infra"
    assert "NVIDIA" in processed.entities
    assert processed.priority_score > 0.0
    assert processed.title_ko
    assert processed.title_en
    assert processed.summary_ko
    assert processed.summary_en
    assert processed.why_it_matters_ko
    assert processed.why_it_matters_en

