from ai_newsletter.clustering import (
    cluster_articles,
    select_primary_article,
    extract_clustering_features,
    is_candidate_compatible_with_cluster,
    _sort_article_indices_for_clustering,
    DEFAULT_EVENT_WINDOW_HOURS,
    DEFAULT_SIMILARITY_THRESHOLD,
    MIN_CLUSTER_COHESION_RATIO,
)
from ai_newsletter.models import Article, Event


def test_cluster_articles_empty():
    events, event_articles, updated = cluster_articles([])
    assert events == []
    assert event_articles == []
    assert updated == []


def test_cluster_articles_grouping():
    a1 = Article(
        title="OpenAI announces GPT-4o with omni modal intelligence",
        url="https://openai.com/index/hello-gpt-4o",
        source="OpenAI Newsroom",
        category="frontier_models",
        is_official=True,
        source_type="official",
        score=95.0,
        tags=["OpenAI"],
    )
    a2 = Article(
        title="OpenAI launches GPT-4o multimodal model for all users",
        url="https://techcrunch.com/openai-gpt4o",
        source="TechCrunch",
        category="frontier_models",
        is_official=False,
        source_type="media",
        score=85.0,
        tags=["OpenAI"],
    )
    a3 = Article(
        title="NVIDIA announces Blackwell B200 shipments for AI servers",
        url="https://nvidianews.nvidia.com/blackwell",
        source="NVIDIA",
        category="hardware_infra",
        is_official=True,
        source_type="official",
        score=92.0,
        tags=["NVIDIA"],
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    # Check that a1 was chosen as primary for OpenAI event
    openai_event = next(e for e in events if "GPT-4o" in e.title or "gpt-4o" in e.title.lower())
    assert openai_event.source_count == 2
    assert openai_event.has_official_source is True
    assert openai_event.official_source_url == "https://openai.com/index/hello-gpt-4o"


def test_select_primary_article():
    official = Article(
        title="Official Announcement",
        url="https://official.com",
        source="OpenAI",
        is_official=True,
        source_type="official",
        score=90.0,
    )
    media = Article(
        title="Media Reporting",
        url="https://media.com",
        source="TechCrunch",
        is_official=False,
        source_type="media",
        score=85.0,
    )
    chosen = select_primary_article([media, official])
    assert chosen == official


# ── Regression Tests for Layered Event Clustering ────────────────────────────


def test_clustering_false_negative_cross_publisher_same_event():
    """False negative: Same event, slightly different wording, different publishers clusters together."""
    a1 = Article(
        title="OpenAI and Anthropic announce new AI safety initiative",
        url="https://techcrunch.com/openai-anthropic-safety",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "Anthropic"],
    )
    a2 = Article(
        title="OpenAI announces partnership with Anthropic on AI safety",
        url="https://reuters.com/openai-anthropic-partnership",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI", "Anthropic"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 1, f"Expected 1 joint event, got {len(events)}"
    assert events[0].source_count == 2
    assert "safety" in events[0].title.lower() or "ai" in events[0].title.lower()


def test_clustering_same_company_different_events_remain_separate():
    """Same company, different event: Must NOT be merged merely because they share a company name."""
    a1 = Article(
        title="OpenAI releases GPT-4o flagship model with multimodal capabilities",
        url="https://openai.com/gpt-4o",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a2 = Article(
        title="OpenAI opens new corporate office in London for European expansion",
        url="https://theverge.com/openai-london-office",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 2, f"Expected 2 separate events, got {len(events)}"


def test_clustering_same_model_different_events_remain_separate():
    """Same model, different event: Release vs Lawsuit clashing actions must remain separate."""
    a1 = Article(
        title="OpenAI launches GPT-4o voice mode for mobile subscribers",
        url="https://techcrunch.com/openai-gpt4o-voice",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "GPT-4o"],
    )
    a2 = Article(
        title="OpenAI sued over GPT-4o training data copyright infringement",
        url="https://reuters.com/openai-gpt4o-lawsuit",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI", "GPT-4o"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 2, f"Expected 2 separate events for release vs lawsuit, got {len(events)}"


def test_clustering_different_models_same_broader_event():
    """Different models, same broader event: Joint safety framework/testing clusters together."""
    a1 = Article(
        title="US AI Safety Institute to evaluate OpenAI GPT-4o and Anthropic Claude 3 models",
        url="https://techcrunch.com/us-aisi-testing",
        source="TechCrunch",
        category="frontier_models",
        tags=["US NIST AISI", "OpenAI", "Anthropic"],
    )
    a2 = Article(
        title="US AISI announces joint safety evaluations for frontier models with Anthropic and OpenAI",
        url="https://reuters.com/us-aisi-safety-evaluation",
        source="Reuters",
        category="frontier_models",
        tags=["US NIST AISI", "OpenAI", "Anthropic"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 1, f"Expected 1 broader event cluster, got {len(events)}"
    assert events[0].source_count == 2


def test_clustering_multilingual_korean_english_coverage():
    """Multilingual coverage: Korean and English reporting of the same event cluster together."""
    a_en = Article(
        title="OpenAI announces GPT-4o flagship model with omni modal intelligence",
        title_ko="오픈AI, 플래그십 AI 모델 GPT-4o 발표",
        url="https://openai.com/gpt-4o",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-4o"],
    )
    a_ko = Article(
        title="오픈AI, 음성 및 비전 지원하는 플래그십 AI 모델 GPT-4o 공개",
        title_en="OpenAI launches flagship AI model GPT-4o with voice and vision",
        url="https://zdnet.co.kr/news/gpt4o-launch",
        source="ZDNet Korea",
        category="frontier_models",
        tags=["OpenAI", "GPT-4o"],
    )

    events, event_articles, updated = cluster_articles([a_en, a_ko])
    assert len(events) == 1, f"Expected 1 bilingual cluster, got {len(events)}"
    assert events[0].source_count == 2


# ── STEP 8.5 Regression Tests: False-Positive & Chaining Hardening ────────────


def test_req1_same_company_different_events_three_distinct():
    """1. Same company, different events: GPT-5 launch vs legacy API pricing vs London office."""
    a1 = Article(
        title="OpenAI announces GPT-5",
        url="https://openai.com/gpt-5",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a2 = Article(
        title="OpenAI announces API pricing and subscription tiers for legacy models",
        url="https://theverge.com/gpt-pricing",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a3 = Article(
        title="OpenAI opens new London office for European team",
        url="https://techcrunch.com/openai-london",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3])
    assert len(events) == 3, f"Expected 3 distinct events, got {len(events)}"


def test_req2_same_model_different_actions_three_distinct():
    """2. Same company, different actions: Release vs Legacy Pricing vs Security vulnerability."""
    a1 = Article(
        title="OpenAI releases GPT-5 flagship model",
        url="https://openai.com/gpt-5-release",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    a2 = Article(
        title="OpenAI changes GPT-3.5 pricing for legacy API developer accounts",
        url="https://techcrunch.com/gpt35-price-change",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "GPT-3.5"],
    )
    a3 = Article(
        title="Researchers discover a GPT-5 safety vulnerability in jailbreak test",
        url="https://wired.com/gpt5-vulnerability",
        source="WIRED",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3])
    assert len(events) == 3, f"Expected 3 distinct events for release, pricing, vulnerability, got {len(events)}"


def test_req3_release_vs_regulatory_investigation():
    """3. Release vs Regulatory: Product launch vs government investigation remain separate."""
    a1 = Article(
        title="OpenAI launches GPT-5 frontier intelligence system",
        url="https://openai.com/gpt-5",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    a2 = Article(
        title="EU investigates OpenAI over AI regulation compliance and data privacy",
        url="https://reuters.com/eu-openai-probe",
        source="Reuters",
        category="regulation_policy",
        tags=["OpenAI", "EU AI Office"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 2, f"Expected 2 separate events, got {len(events)}"


def test_req4_corporate_expansion_vs_model_release():
    """4. Corporate expansion vs Model release remain separate."""
    a1 = Article(
        title="Anthropic opens a new London office for European operations",
        url="https://techcrunch.com/anthropic-london",
        source="TechCrunch",
        category="frontier_models",
        tags=["Anthropic"],
    )
    a2 = Article(
        title="Anthropic releases Claude update with extended context window",
        url="https://anthropic.com/claude-update",
        source="Anthropic",
        category="frontier_models",
        tags=["Anthropic"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 2, f"Expected 2 separate events, got {len(events)}"


def test_req5_partnership_vs_product_release():
    """5. Partnership vs Product release remain separate."""
    a1 = Article(
        title="OpenAI launches GPT-5 flagship model for public use",
        url="https://openai.com/gpt-5-launch",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    a2 = Article(
        title="OpenAI partners with Microsoft on datacenter infrastructure expansion",
        url="https://bloomberg.com/openai-msft-infrastructure",
        source="Bloomberg",
        category="frontier_models",
        tags=["OpenAI", "Microsoft"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 2, f"Expected 2 separate events, got {len(events)}"


def test_req6_legitimate_same_event_cross_publisher_clustering():
    """6. Legitimate cross-publisher coverage of same event produces 1 event with 3 publishers."""
    a1 = Article(
        title="OpenAI announces GPT-5",
        url="https://openai.com/index/gpt-5",
        source="OpenAI",
        publisher="OpenAI",
        is_official=True,
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    a2 = Article(
        title="OpenAI launches GPT-5 with new capabilities",
        url="https://reuters.com/technology/openai-gpt5",
        source="Reuters",
        publisher="Reuters",
        is_official=False,
        source_type="media",
        evidence_level="independent",
        is_independent_source=True,
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    a3 = Article(
        title="OpenAI's GPT-5 arrives with major improvements",
        url="https://techcrunch.com/openai-gpt5-launch",
        source="TechCrunch",
        publisher="TechCrunch",
        is_official=False,
        source_type="media",
        evidence_level="industry_media",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3])
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"
    ev = events[0]
    assert ev.source_count == 3
    assert ev.independent_source_count == 1
    assert ev.has_official_source is True


def test_req7_legitimate_same_event_different_wording_clustering():
    """7. Legitimate same event with substantially different wording clusters together."""
    a1 = Article(
        title="OpenAI announces a new frontier model",
        title_ko="오픈AI, 새로운 프론티어 모델 발표",
        excerpt="OpenAI has announced its next generation frontier AI system.",
        url="https://openai.com/new-model",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a2 = Article(
        title="New OpenAI model raises the bar for AI performance",
        title_ko="새로운 오픈AI 모델, AI 성능 기준을 높이다",
        excerpt="The latest AI system from OpenAI achieves benchmark records.",
        url="https://reuters.com/new-openai-model",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a3 = Article(
        title="OpenAI unveils its latest flagship AI system",
        title_ko="오픈AI, 최신 플래그십 AI 시스템 공개",
        excerpt="OpenAI unveiled its flagship frontier AI system today.",
        url="https://techcrunch.com/openai-flagship-system",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3])
    assert len(events) == 1, f"Expected 1 event for differently-worded coverage, got {len(events)}"
    assert events[0].source_count == 3


def test_req8_legitimate_broader_event_with_different_model_references():
    """8. Legitimate broader event: Joint safety testing across multiple models clusters together."""
    a1 = Article(
        title="OpenAI and Anthropic announce joint AI safety initiative with US AISI",
        url="https://openai.com/safety-initiative",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "Anthropic", "US NIST AISI"],
    )
    a2 = Article(
        title="OpenAI tests GPT-4o under new US AISI safety evaluation framework",
        url="https://techcrunch.com/openai-aisi",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "US NIST AISI"],
    )
    a3 = Article(
        title="Anthropic tests Claude 3.5 in US AISI joint safety evaluation pact",
        url="https://reuters.com/anthropic-aisi",
        source="Reuters",
        category="frontier_models",
        tags=["Anthropic", "US NIST AISI"],
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3])
    assert len(events) == 1, f"Expected 1 joint event cluster, got {len(events)}"
    assert events[0].source_count == 3


def test_req9_single_linkage_chaining_safeguard_a_b_c():
    """9. A-B-C chaining safeguard: A and C are clearly unrelated and must NOT be in the same cluster."""
    # A: Pure model release
    a = Article(
        article_id="art_a",
        title="OpenAI announces GPT-5 frontier model",
        url="https://openai.com/gpt-5-announcement",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    # B: Broad article mentioning GPT-5 and London compute deployment
    b = Article(
        article_id="art_b",
        title="OpenAI discusses GPT-5 rollout plans and European infrastructure in London",
        url="https://theverge.com/openai-europe-gpt5",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    # C: Pure corporate office opening for sales
    c = Article(
        article_id="art_c",
        title="OpenAI opens new corporate office in London for sales operations",
        url="https://techcrunch.com/openai-london-sales",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected exactly 2 events (A+B and C), got {len(events)}"

    cluster_a = next(e for e in events if a.article_id in [art.article_id for art in updated if art.event_id == e.event_id])
    cluster_c = next(e for e in events if c.article_id in [art.article_id for art in updated if art.event_id == e.event_id])

    assert cluster_a.event_id != cluster_c.event_id
    articles_in_a = [art.article_id for art in updated if art.event_id == cluster_a.event_id]
    articles_in_c = [art.article_id for art in updated if art.event_id == cluster_c.event_id]

    assert set(articles_in_a) == {"art_a", "art_b"}
    assert set(articles_in_c) == {"art_c"}


def test_bridge_a_release_bridge_office_expansion():
    """Bridge Test A: Launch story (A) and bridge story (B) cluster together; office opening (C) stays separate."""
    a = Article(
        article_id="art_a_launch",
        title="OpenAI announces GPT-5 frontier AI model",
        url="https://openai.com/gpt-5-announcement",
        source="OpenAI",
        is_official=True,
        source_type="official",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="art_b_bridge",
        title="OpenAI announces GPT-5 model availability across European London offices",
        url="https://theverge.com/openai-europe-gpt5",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    c = Article(
        article_id="art_c_office",
        title="OpenAI opens new corporate office in London for sales operations",
        url="https://techcrunch.com/openai-london-sales",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    a_ev = next(art.event_id for art in updated if art.article_id == "art_a_launch")
    b_ev = next(art.event_id for art in updated if art.article_id == "art_b_bridge")
    c_ev = next(art.event_id for art in updated if art.article_id == "art_c_office")

    assert a_ev == b_ev, "Article A and Bridge B should cluster together"
    assert c_ev != a_ev, "Office opening C must NOT cluster with launch event"


def test_bridge_b_release_bridge_legal():
    """Bridge Test B: Launch story (A) and bridge story (B) cluster; pure regulatory investigation (C) stays separate."""
    a = Article(
        article_id="art_a_release",
        title="OpenAI announces GPT-5 frontier AI model",
        url="https://openai.com/gpt-5-announcement",
        source="OpenAI",
        is_official=True,
        source_type="official",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="art_b_bridge",
        title="OpenAI announces GPT-5 release amid European regulatory developments",
        url="https://theverge.com/openai-europe-gpt5-reg",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    c = Article(
        article_id="art_c_probe",
        title="EU opens formal antitrust probe and investigation into OpenAI",
        url="https://reuters.com/eu-openai-probe",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    a_ev = next(art.event_id for art in updated if art.article_id == "art_a_release")
    b_ev = next(art.event_id for art in updated if art.article_id == "art_b_bridge")
    c_ev = next(art.event_id for art in updated if art.article_id == "art_c_probe")

    assert a_ev == b_ev, "Article A and Bridge B should cluster together"
    assert c_ev != a_ev, "EU antitrust probe C must NOT cluster with model launch event"


def test_bridge_c_pricing_bridge_release():
    """Bridge Test C: Launch story (A) vs Unrelated pricing announcement (B) must remain cleanly separated."""
    a = Article(
        article_id="art_a_launch",
        title="OpenAI launches GPT-5 flagship model for general availability",
        url="https://openai.com/gpt-5-launch",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="art_b_pricing",
        title="OpenAI introduces new subscription pricing and API rates for legacy GPT-3.5 models",
        url="https://techcrunch.com/openai-gpt-pricing",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "GPT-3.5"],
    )

    events, event_articles, updated = cluster_articles([a, b])
    assert len(events) == 2, f"Expected 2 separate events for launch vs pricing, got {len(events)}"


def test_bridge_d_two_unrelated_corporate_events():
    """Bridge Test D: Office opening (A) vs Executive appointment (B) must remain separate."""
    a = Article(
        article_id="art_a_office",
        title="Anthropic opens new corporate headquarters and office in London",
        url="https://techcrunch.com/anthropic-london",
        source="TechCrunch",
        category="frontier_models",
        tags=["Anthropic"],
    )
    b = Article(
        article_id="art_b_exec",
        title="Anthropic appoints new chief operating officer and executive hire",
        url="https://reuters.com/anthropic-exec",
        source="Reuters",
        category="frontier_models",
        tags=["Anthropic"],
    )

    events, event_articles, updated = cluster_articles([a, b])
    assert len(events) == 2, f"Expected 2 separate events for office vs executive hire, got {len(events)}"


def test_req10_evidence_metadata_integrity_after_clustering():
    """10. Verify evidence metadata (source_count, independent_source_count, status, confidence) remains correct."""
    a1 = Article(
        title="OpenAI announces GPT-5 flagship model",
        url="https://openai.com/gpt-5",
        source="OpenAI",
        publisher="OpenAI",
        is_official=True,
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        priority_score=95.0,
    )
    a2 = Article(
        title="OpenAI launches GPT-5 with multimodal advances",
        url="https://reuters.com/gpt-5",
        source="Reuters",
        publisher="Reuters",
        is_official=False,
        source_type="media",
        evidence_level="independent",
        is_independent_source=True,
        priority_score=90.0,
    )
    a3 = Article(
        title="OpenAI's GPT-5 debuts across all major platforms",
        url="https://techcrunch.com/gpt-5",
        source="TechCrunch",
        publisher="TechCrunch",
        is_official=False,
        source_type="media",
        evidence_level="industry_media",
        priority_score=85.0,
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3])
    assert len(events) == 1
    ev = events[0]
    assert ev.source_count == 3
    assert ev.independent_source_count == 1
    assert ev.evidence_diversity == 3
    assert ev.verification_status == "independently_reported"
    assert ev.primary_sources == ["openai"]
    assert ev.independent_sources == ["reuters"]
    assert ev.evidence_sources == ["openai", "reuters", "techcrunch"]
    assert ev.confidence_score >= 70.0
    assert ev.confidence_label == "High"

    # Verify article fields updated consistently
    for art in updated:
        assert art.event_id == ev.event_id
        assert art.event_source_count == 3
        assert art.event_independent_source_count == 1
        assert art.event_verification_status == "independently_reported"
        assert art.event_confidence_label == "High"


# ── STEP 8.7 Regression Tests: Event Anchor & Representative Hardening ────────


def test_step87_test_a_classic_chaining_three_articles():
    """Test A: Classic chaining (A launch, B broad bridge, C office opening) -> 2 events ([A, B] and [C])."""
    a = Article(
        article_id="step87_a_launch",
        title="OpenAI announces GPT-5 frontier AI model",
        url="https://openai.com/gpt-5-announcement",
        source="OpenAI",
        is_official=True,
        source_type="official",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step87_b_bridge",
        title="OpenAI announces GPT-5 model availability across European London offices",
        url="https://theverge.com/openai-europe-gpt5",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    c = Article(
        article_id="step87_c_office",
        title="OpenAI opens new corporate office in London for sales operations",
        url="https://techcrunch.com/openai-london-sales",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    a_ev = next(art.event_id for art in updated if art.article_id == "step87_a_launch")
    b_ev = next(art.event_id for art in updated if art.article_id == "step87_b_bridge")
    c_ev = next(art.event_id for art in updated if art.article_id == "step87_c_office")

    assert a_ev == b_ev, "Article A and Bridge B should cluster together"
    assert c_ev != a_ev, "Office opening C must NOT cluster with launch event"


def test_step87_test_b_release_broad_office_expansion():
    """Test B: Release -> Broad -> Office expansion -> 2 separate events."""
    a = Article(
        article_id="step87_b_rel",
        title="Anthropic releases Claude 3.5 Sonnet frontier model",
        url="https://anthropic.com/claude-3-5-sonnet",
        source="Anthropic",
        is_official=True,
        source_type="official",
        category="frontier_models",
        tags=["Anthropic", "Claude 3.5 Sonnet"],
    )
    b = Article(
        article_id="step87_b_broad",
        title="Anthropic releases Claude 3.5 Sonnet amid expanding European headquarters footprint",
        url="https://theverge.com/anthropic-europe-growth",
        source="The Verge",
        category="frontier_models",
        tags=["Anthropic", "Claude 3.5 Sonnet"],
    )
    c = Article(
        article_id="step87_b_office",
        title="Anthropic opens new corporate headquarters and office in Dublin",
        url="https://techcrunch.com/anthropic-dublin-office",
        source="TechCrunch",
        category="frontier_models",
        tags=["Anthropic"],
    )

    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    a_ev = next(art.event_id for art in updated if art.article_id == "step87_b_rel")
    b_ev = next(art.event_id for art in updated if art.article_id == "step87_b_broad")
    c_ev = next(art.event_id for art in updated if art.article_id == "step87_b_office")

    assert a_ev == b_ev
    assert c_ev != a_ev


def test_step87_test_c_release_broad_legal():
    """Test C: Release -> Broad -> Legal action -> 2 separate events."""
    a = Article(
        article_id="step87_c_rel",
        title="OpenAI announces GPT-5 frontier AI model",
        url="https://openai.com/gpt-5-announcement",
        source="OpenAI",
        is_official=True,
        source_type="official",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step87_c_broad",
        title="OpenAI announces GPT-5 release amid European regulatory developments",
        url="https://theverge.com/openai-europe-gpt5-reg",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    c = Article(
        article_id="step87_c_legal",
        title="EU opens formal antitrust probe and investigation into OpenAI",
        url="https://reuters.com/eu-openai-probe",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    a_ev = next(art.event_id for art in updated if art.article_id == "step87_c_rel")
    b_ev = next(art.event_id for art in updated if art.article_id == "step87_c_broad")
    c_ev = next(art.event_id for art in updated if art.article_id == "step87_c_legal")

    assert a_ev == b_ev
    assert c_ev != a_ev


def test_step87_test_d_pricing_broad_release():
    """Test D: Pricing -> Broad -> Release (Unrelated pricing and release stories stay separate)."""
    # Distinct pricing vs release stories stay separate without joint release launch anchor
    a = Article(
        article_id="step87_d_launch",
        title="OpenAI launches GPT-5 frontier AI model for all users",
        url="https://openai.com/gpt-5-launch",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step87_d_pricing",
        title="OpenAI introduces new subscription pricing and API rates for legacy older systems",
        url="https://techcrunch.com/openai-legacy-pricing",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )

    events, event_articles, updated = cluster_articles([a, b])
    assert len(events) == 2, f"Expected 2 separate events for launch vs pricing, got {len(events)}"


def test_step87_test_e_three_unrelated_corporate_events():
    """Test E: Unrelated corporate events (office expansion, investment, legal action -> 3 events)."""
    a = Article(
        article_id="step87_e_office",
        title="Google opens new corporate headquarters and office in London",
        url="https://theverge.com/google-london-office",
        source="The Verge",
        category="frontier_models",
        tags=["Google"],
    )
    b = Article(
        article_id="step87_e_funding",
        title="Google raises funding and investment series for European startups",
        url="https://techcrunch.com/google-europe-funding",
        source="TechCrunch",
        category="frontier_models",
        tags=["Google"],
    )
    c = Article(
        article_id="step87_e_legal",
        title="EU court issues antitrust fine and regulatory lawsuit against Google",
        url="https://reuters.com/google-eu-fine",
        source="Reuters",
        category="frontier_models",
        tags=["Google"],
    )

    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 3, f"Expected 3 separate events, got {len(events)}"


def test_step87_test_f_legitimate_cross_publisher_coverage():
    """Test F: Legitimate cross-publisher same event coverage across 4 distinct publishers -> 1 event."""
    a1 = Article(
        article_id="step87_f_official",
        title="OpenAI announces GPT-5 frontier multimodal AI system",
        url="https://openai.com/gpt-5",
        source="OpenAI",
        publisher="OpenAI",
        is_official=True,
        source_type="official",
        evidence_level="primary",
        is_primary_source=True,
        tags=["OpenAI", "GPT-5"],
    )
    a2 = Article(
        article_id="step87_f_reuters",
        title="OpenAI launches GPT-5 with major multimodal intelligence breakthroughs",
        url="https://reuters.com/openai-gpt5",
        source="Reuters",
        publisher="Reuters",
        is_official=False,
        source_type="media",
        evidence_level="independent",
        is_independent_source=True,
        tags=["OpenAI", "GPT-5"],
    )
    a3 = Article(
        article_id="step87_f_bloomberg",
        title="OpenAI debuts GPT-5 frontier AI model for enterprise and consumers",
        url="https://bloomberg.com/openai-gpt5",
        source="Bloomberg",
        publisher="Bloomberg",
        is_official=False,
        source_type="media",
        evidence_level="independent",
        is_independent_source=True,
        tags=["OpenAI", "GPT-5"],
    )
    a4 = Article(
        article_id="step87_f_techcrunch",
        title="OpenAI's GPT-5 launches with advanced reasoning capabilities",
        url="https://techcrunch.com/openai-gpt5-launch",
        source="TechCrunch",
        publisher="TechCrunch",
        is_official=False,
        source_type="media",
        evidence_level="industry_media",
        tags=["OpenAI", "GPT-5"],
    )

    events, event_articles, updated = cluster_articles([a1, a2, a3, a4])
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"
    ev = events[0]
    assert ev.source_count == 4
    assert ev.independent_source_count == 2
    assert ev.has_official_source is True
    assert ev.verification_status == "multi_source"


def test_step87_test_g_broader_multi_entity_safety_initiative():
    """Test G: Broader multi-entity safety initiative across OpenAI and Anthropic -> 1 event."""
    a1 = Article(
        article_id="step87_g_pact",
        title="OpenAI and Anthropic announce joint AI safety initiative and pact",
        url="https://techcrunch.com/openai-anthropic-safety-pact",
        source="TechCrunch",
        publisher="TechCrunch",
        evidence_level="independent",
        tags=["OpenAI", "Anthropic"],
    )
    a2 = Article(
        article_id="step87_g_reuters",
        title="Anthropic announces partnership with OpenAI on frontier model safety evaluations",
        url="https://reuters.com/anthropic-openai-safety",
        source="Reuters",
        publisher="Reuters",
        evidence_level="independent",
        tags=["OpenAI", "Anthropic"],
    )

    events, event_articles, updated = cluster_articles([a1, a2])
    assert len(events) == 1, f"Expected 1 joint event, got {len(events)}"
    assert events[0].source_count == 2
    assert "safety" in events[0].title.lower() or "openai" in events[0].title.lower()


# ── STEP 8.9 Tests: Determinism & Performance Scaling ──────────────────────────

def test_step89_clustering_determinism():
    """Verify that repeated clustering runs produce bit-for-bit identical results."""
    articles = [
        Article(
            article_id=f"det_{i}",
            title=f"OpenAI announces GPT-5 frontier AI release {i}",
            url=f"https://openai.com/gpt-5-det-{i}",
            source="OpenAI" if i == 0 else f"Media{i}",
            publisher="OpenAI" if i == 0 else f"Publisher{i}",
            is_official=(i == 0),
            source_type="official" if i == 0 else "media",
            evidence_level="primary" if i == 0 else "independent",
            tags=["OpenAI", "GPT-5"],
        )
        for i in range(5)
    ] + [
        Article(
            article_id=f"anthropic_{i}",
            title=f"Anthropic opens new corporate headquarters office in London {i}",
            url=f"https://anthropic.com/london-{i}",
            source=f"Source{i}",
            publisher=f"PublisherAnthropic{i}",
            category="frontier_models",
            tags=["Anthropic"],
        )
        for i in range(3)
    ]

    # Run 1
    events1, ea1, updated1 = cluster_articles(articles)

    # Run 2 with same order
    events2, ea2, updated2 = cluster_articles(articles)

    # Run 3 with reversed input order (determinism test)
    events3, ea3, updated3 = cluster_articles(list(reversed(articles)))

    # Verify event counts
    assert len(events1) == len(events2) == len(events3) == 2

    # Verify event IDs and primary articles match exactly
    ev_ids_1 = sorted(e.event_id for e in events1)
    ev_ids_2 = sorted(e.event_id for e in events2)
    ev_ids_3 = sorted(e.event_id for e in events3)
    assert ev_ids_1 == ev_ids_2 == ev_ids_3

    # Verify cluster article groupings match
    grouping_1 = {e.event_id: sorted([a.article_id for a in updated1 if a.event_id == e.event_id]) for e in events1}
    grouping_2 = {e.event_id: sorted([a.article_id for a in updated2 if a.event_id == e.event_id]) for e in events2}
    grouping_3 = {e.event_id: sorted([a.article_id for a in updated3 if a.event_id == e.event_id]) for e in events3}

    assert grouping_1 == grouping_2 == grouping_3


def test_step89_clustering_performance_scaling():
    """Verify performance scaling across synthetic datasets of 100, 300, 500, and 1000 articles without abnormal blow-ups."""
    import time

    def generate_synthetic_articles(count: int) -> list[Article]:
        topics = [
            ("OpenAI", "GPT-5", "announces flagship model release and benchmarks"),
            ("Anthropic", "Claude 3.5 Sonnet", "introduces updated enterprise pricing and rate tiers"),
            ("Google", "Gemini 2.0", "unveils multimodal capabilities for developers"),
            ("Meta", "Llama 3.3", "releases open weights foundation model"),
            ("NVIDIA", "Blackwell B200", "begins volume production and datacenter shipments"),
            ("Mistral", "Le Chat", "opens new corporate office in London headquarters"),
            ("Microsoft", "Copilot", "faces European regulatory investigation and antitrust lawsuit"),
            ("DeepSeek", "DeepSeek R1", "publishes breakthrough reasoning benchmark results"),
        ]
        articles = []
        for i in range(count):
            topic = topics[i % len(topics)]
            pub_idx = (i // len(topics))
            articles.append(
                Article(
                    article_id=f"synth_{count}_{i}",
                    title=f"{topic[0]} {topic[1]} {topic[2]} - variant #{i}",
                    url=f"https://synthetic-news.example.com/art/{count}/{i}",
                    source=f"Publisher_{pub_idx}",
                    publisher=f"Publisher_{pub_idx}",
                    category="frontier_models",
                    tags=[topic[0], topic[1]],
                    excerpt=f"Full coverage of {topic[0]} and {topic[1]} with detail number {i}.",
                    priority_score=50.0 + (i % 50),
                )
            )
        return articles

    timings = {}
    for n in [100, 300, 500, 1000]:
        arts = generate_synthetic_articles(n)
        start_time = time.perf_counter()
        events, ea, updated = cluster_articles(arts)
        elapsed = time.perf_counter() - start_time
        timings[n] = elapsed

        # Assert correct integrity
        assert len(updated) == n
        assert len(events) >= 1
        assert len(ea) == n

    # Verify scaling characteristics without brittle absolute machine-specific timing:
    # 1. Processing 1000 articles must be greater than or equal to 100 articles
    assert timings[1000] >= timings[100]
    # 2. Scaling ratio between 10x dataset growth (100 -> 1000) must scale smoothly without abnormal blow-ups (< 100x ratio)
    scaling_ratio = timings[1000] / max(timings[100], 0.001)
    assert scaling_ratio < 100.0, f"Abnormal non-linear blowup detected: 10x size gave {scaling_ratio:.1f}x time"


# ── STEP 8.7.1 Mandatory Regression Tests ─────────────────────────────────────


def test_step871_test_a_same_launch_pricing_and_release():
    """Test A: Same launch pricing + release MUST assert len(events) == 1 and same event_id."""
    a = Article(
        article_id="step871_a1_launch",
        title="OpenAI launches GPT-5 frontier AI model",
        url="https://openai.com/gpt-5-launch",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step871_a2_pricing",
        title="OpenAI announces GPT-5 pricing and API rates",
        url="https://theverge.com/openai-gpt-5-pricing",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    events, event_articles, updated = cluster_articles([a, b])
    assert len(events) == 1, f"Expected 1 event for same launch pricing + release, got {len(events)}"
    a_ev = next(art.event_id for art in updated if art.article_id == "step871_a1_launch")
    b_ev = next(art.event_id for art in updated if art.article_id == "step871_a2_pricing")
    assert a_ev == b_ev, "Both article IDs must have the same event_id"
    assert a_ev == events[0].event_id


def test_step871_test_b_unrelated_pricing_and_release():
    """Test B: Unrelated pricing + release MUST assert len(events) == 2."""
    a = Article(
        article_id="step871_b1_launch",
        title="OpenAI launches GPT-5",
        url="https://openai.com/gpt-5-launch",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step871_b2_pricing",
        title="OpenAI changes API pricing for an unrelated older product",
        url="https://techcrunch.com/openai-older-pricing",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI"],
    )
    events, event_articles, updated = cluster_articles([a, b])
    assert len(events) == 2, f"Expected 2 events for unrelated pricing + release, got {len(events)}"
    a_ev = next(art.event_id for art in updated if art.article_id == "step871_b1_launch")
    b_ev = next(art.event_id for art in updated if art.article_id == "step871_b2_pricing")
    assert a_ev != b_ev, "Unrelated pricing and release must have distinct event_ids"


def test_step871_test_c_exact_chaining_membership():
    """Test C: Exact chaining membership: A = release, B = release + broad office wording, C = office."""
    a = Article(
        article_id="step871_c_rel",
        title="Anthropic releases Claude 3.5 Sonnet frontier AI model",
        url="https://anthropic.com/claude-35-sonnet",
        source="Anthropic",
        category="frontier_models",
        tags=["Anthropic", "Claude 3.5 Sonnet"],
    )
    b = Article(
        article_id="step871_c_broad",
        title="Anthropic releases Claude 3.5 Sonnet model availability amid European headquarters expansion",
        url="https://theverge.com/anthropic-claude-35-europe",
        source="The Verge",
        category="frontier_models",
        tags=["Anthropic", "Claude 3.5 Sonnet"],
    )
    c = Article(
        article_id="step871_c_office",
        title="Anthropic opens new corporate office in London for sales operations",
        url="https://techcrunch.com/anthropic-london-office",
        source="TechCrunch",
        category="frontier_models",
        tags=["Anthropic"],
    )
    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    cluster_memberships = [
        {art.article_id for art in updated if art.event_id == ev.event_id}
        for ev in events
    ]
    assert {"step871_c_rel", "step871_c_broad"} in cluster_memberships, "Expected {A, B} exact cluster"
    assert {"step871_c_office"} in cluster_memberships, "Expected {C} exact cluster"


def test_step871_test_d_conflict_loophole_not_bypassed():
    """Test D: Conflict loophole - shared action does NOT automatically bypass ACTION_CONFLICTS."""
    a = Article(
        article_id="step871_d_rel_part",
        title="OpenAI and Microsoft partner to release new frontier AI model",
        url="https://openai.com/msft-partnership-release",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "Microsoft"],
    )
    b = Article(
        article_id="step871_d_part",
        title="OpenAI and Microsoft announce strategic partnership expansion",
        url="https://theverge.com/msft-openai-partnership",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "Microsoft"],
    )
    c = Article(
        article_id="step871_d_legal",
        title="EU opens formal antitrust lawsuit and regulatory investigation into OpenAI",
        url="https://reuters.com/eu-openai-lawsuit",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI"],
    )
    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"
    a_ev = next(art.event_id for art in updated if art.article_id == "step871_d_rel_part")
    b_ev = next(art.event_id for art in updated if art.article_id == "step871_d_part")
    c_ev = next(art.event_id for art in updated if art.article_id == "step871_d_legal")
    assert a_ev == b_ev, "A and B should cluster together into the partnership event"
    assert c_ev != a_ev, "EU regulatory lawsuit C must NOT bypass action conflicts and merge into release/partnership"


def test_step871_test_e_structured_event_anchor():
    """Test E: Structured EventAnchor compatibility.
    (same entity + same model + compatible actions) gets stronger compatibility than
    (same entity + different model + unrelated action).
    """
    a = Article(
        article_id="step871_e_rel",
        title="OpenAI launches GPT-5 flagship model for enterprise developers",
        url="https://openai.com/gpt5-enterprise",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step871_e_pricing",
        title="OpenAI announces GPT-5 pricing and token costs for developers",
        url="https://theverge.com/gpt5-token-pricing",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    c = Article(
        article_id="step871_e_unrelated",
        title="Researchers discover security vulnerability in legacy GPT-3.5 safety filters",
        url="https://wired.com/gpt35-vulnerability",
        source="WIRED",
        category="frontier_models",
        tags=["OpenAI", "GPT-3.5"],
    )
    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"
    a_ev = next(art.event_id for art in updated if art.article_id == "step871_e_rel")
    b_ev = next(art.event_id for art in updated if art.article_id == "step871_e_pricing")
    c_ev = next(art.event_id for art in updated if art.article_id == "step871_e_unrelated")
    assert a_ev == b_ev, "Same entity + same model + compatible actions must cluster"
    assert c_ev != a_ev, "Different model + unrelated action must stay separate"


# ── STEP 8.7.2 Mandatory Regression Tests: Structured EventAnchor ─────────────


def test_step872_test_a_structured_same_launch():
    """Test A: (release, openai, gpt5) and (pricing, openai, gpt5) -> One event, same event_id."""
    a = Article(
        article_id="step872_a_rel",
        title="OpenAI releases GPT-5 frontier model",
        url="https://openai.com/gpt-5-release",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step872_a_pricing",
        title="OpenAI announces GPT-5 pricing and API subscription tiers",
        url="https://theverge.com/gpt-5-pricing-tiers",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    events, event_articles, updated = cluster_articles([a, b])
    assert len(events) == 1, f"Expected 1 event for structured same launch, got {len(events)}"
    a_ev = next(art.event_id for art in updated if art.article_id == "step872_a_rel")
    b_ev = next(art.event_id for art in updated if art.article_id == "step872_a_pricing")
    assert a_ev == b_ev, "Both articles must share the same event_id"
    assert a_ev == events[0].event_id


def test_step872_test_b_structured_different_model():
    """Test B: (release, openai, gpt5) vs (pricing, openai, claude) -> Separate events."""
    a = Article(
        article_id="step872_b_rel",
        title="OpenAI releases GPT-5 frontier model",
        url="https://openai.com/gpt-5-release",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step872_b_pricing",
        title="OpenAI announces Claude model API pricing update",
        url="https://techcrunch.com/openai-claude-pricing",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "Claude"],
    )
    events, event_articles, updated = cluster_articles([a, b])
    assert len(events) == 2, f"Expected 2 separate events for different structured model anchors, got {len(events)}"
    a_ev = next(art.event_id for art in updated if art.article_id == "step872_b_rel")
    b_ev = next(art.event_id for art in updated if art.article_id == "step872_b_pricing")
    assert a_ev != b_ev, "Articles with different structured models must have distinct event_ids"


def test_step872_test_c_unrelated_additional_model():
    """Test C: Pricing mentions GPT-5. Release mentions GPT-5 and unrelated second model (Claude).
    Ensures implementation does not rely on an unsafe one-sided subset check (pricing_models <= release_models).
    """
    a_pricing = Article(
        article_id="step872_c_pricing",
        title="OpenAI announces GPT-5 pricing and API token rates",
        url="https://theverge.com/openai-gpt5-pricing",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b_release_multi = Article(
        article_id="step872_c_rel_multi",
        title="OpenAI launches GPT-5 model alongside Anthropic Claude comparison benchmark",
        url="https://techcrunch.com/openai-gpt5-claude-launch",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "GPT-5", "Claude"],
    )
    events, event_articles, updated = cluster_articles([a_pricing, b_release_multi])
    assert len(events) == 2, f"Expected 2 separate events due to unrelated additional model in release, got {len(events)}"
    p_ev = next(art.event_id for art in updated if art.article_id == "step872_c_pricing")
    r_ev = next(art.event_id for art in updated if art.article_id == "step872_c_rel_multi")
    assert p_ev != r_ev, "Unrelated additional model must prevent single launch merge"


def test_step872_test_d_conflict_loophole():
    """Test D: Cluster contains shared action (partnership) plus conflicting action (release).
    Candidate has unrelated conflicting action (legal).
    Shared action (partnership) must NOT bypass the specific release ↔ legal conflict.
    """
    a_partner_rel = Article(
        article_id="step872_d_partner_rel",
        title="OpenAI and Microsoft partner to release new frontier AI model",
        url="https://openai.com/msft-partnership-release",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "Microsoft"],
    )
    b_partner = Article(
        article_id="step872_d_partner",
        title="OpenAI and Microsoft announce strategic partnership expansion",
        url="https://theverge.com/msft-openai-partnership",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "Microsoft"],
    )
    c_legal = Article(
        article_id="step872_d_legal",
        title="EU opens formal antitrust lawsuit and regulatory investigation into OpenAI",
        url="https://reuters.com/eu-openai-lawsuit",
        source="Reuters",
        category="frontier_models",
        tags=["OpenAI"],
    )
    events, event_articles, updated = cluster_articles([a_partner_rel, b_partner, c_legal])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"
    a_ev = next(art.event_id for art in updated if art.article_id == "step872_d_partner_rel")
    b_ev = next(art.event_id for art in updated if art.article_id == "step872_d_partner")
    c_ev = next(art.event_id for art in updated if art.article_id == "step872_d_legal")
    assert a_ev == b_ev, "A and B should cluster into the partnership event"
    assert c_ev != a_ev, "Legal action C must NOT merge into release/partnership cluster despite shared partnership"


def test_step872_test_e_exact_chaining():
    """Test E: Exact chaining membership:
    A = release
    B = release + broad office wording
    C = office
    Assert exact membership: {A, B} and {C}.
    """
    a = Article(
        article_id="step872_e_rel",
        title="Anthropic releases Claude 3.5 Sonnet frontier AI model",
        url="https://anthropic.com/claude-35-sonnet",
        source="Anthropic",
        category="frontier_models",
        tags=["Anthropic", "Claude 3.5 Sonnet"],
    )
    b = Article(
        article_id="step872_e_broad",
        title="Anthropic releases Claude 3.5 Sonnet model availability amid European headquarters expansion",
        url="https://theverge.com/anthropic-claude-35-europe",
        source="The Verge",
        category="frontier_models",
        tags=["Anthropic", "Claude 3.5 Sonnet"],
    )
    c = Article(
        article_id="step872_e_office",
        title="Anthropic opens new corporate office in London for sales operations",
        url="https://techcrunch.com/anthropic-london-office",
        source="TechCrunch",
        category="frontier_models",
        tags=["Anthropic"],
    )
    events, event_articles, updated = cluster_articles([a, b, c])
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    cluster_memberships = [
        {art.article_id for art in updated if art.event_id == ev.event_id}
        for ev in events
    ]
    assert {"step872_e_rel", "step872_e_broad"} in cluster_memberships, "Expected {A, B} exact cluster"
    assert {"step872_e_office"} in cluster_memberships, "Expected {C} exact cluster"


# ── STEP 8.7.3 Mandatory Regression Tests: Aggregate Cluster Pricing Context ───


def test_step873_aggregate_cluster_unrelated_pricing():
    """Test STEP 8.7.3: Aggregate cluster containing both pricing and release actions
    must NOT allow an unrelated Claude pricing article to merge into the GPT-5 launch cluster.

    Cluster:
        A = OpenAI releases GPT-5
        B = OpenAI announces GPT-5 pricing
    Candidate:
        C = OpenAI announces Claude model API pricing update

    Expected:
        GPT-5 release + GPT-5 pricing -> one event {A, B}
        Unrelated Claude pricing -> separate event {C}
    Must be deterministic under both forward and reversed input orders.
    """
    a = Article(
        article_id="step873_a_gpt5_rel",
        title="OpenAI releases GPT-5 frontier AI model",
        url="https://openai.com/gpt5-release",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    b = Article(
        article_id="step873_b_gpt5_pricing",
        title="OpenAI announces GPT-5 pricing and API subscription tiers",
        url="https://theverge.com/gpt5-pricing",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    c = Article(
        article_id="step873_c_claude_pricing",
        title="OpenAI announces Claude model API pricing update",
        url="https://techcrunch.com/claude-pricing",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "Claude"],
    )

    for order_name, article_list in [("forward", [a, b, c]), ("reversed", [c, b, a])]:
        events, event_articles, updated = cluster_articles(article_list)
        assert len(events) == 2, f"[{order_name}] Expected exactly 2 events, got {len(events)}"

        memberships = [
            {art.article_id for art in updated if art.event_id == ev.event_id}
            for ev in events
        ]
        assert {"step873_a_gpt5_rel", "step873_b_gpt5_pricing"} in memberships, (
            f"[{order_name}] Expected GPT-5 release + pricing to form a single exact cluster"
        )
        assert {"step873_c_claude_pricing"} in memberships, (
            f"[{order_name}] Expected unrelated Claude pricing to form an isolated exact cluster"
        )


# ── STEP 8.7.4 / 8.7.5 Permutation Determinism & Fixture Isolation ──────────


def _assert_gpt5_launch_clustering_membership(
    events: list[Event],
    updated: list[Article],
    test_label: str,
    id_a: str,
    id_b: str,
    id_c: str,
) -> None:
    """Verify that clustering produced exactly two events: {A, B} and {C}."""
    assert len(events) == 2, f"[{test_label}] Expected exactly 2 events, got {len(events)}"

    assigned_ids = [art.article_id for art in updated if art.event_id]
    assert len(assigned_ids) == 3, f"[{test_label}] Expected 3 assigned articles, got {len(assigned_ids)}"
    assert set(assigned_ids) == {id_a, id_b, id_c}, (
        f"[{test_label}] Missing articles: {{id_a, id_b, id_c}} - {set(assigned_ids)}"
    )

    ev_by_art = {art.article_id: art.event_id for art in updated}
    assert len(ev_by_art) == 3, f"[{test_label}] Duplicate article assignment detected"

    assert ev_by_art[id_a] == ev_by_art[id_b], (
        f"[{test_label}] A and B must belong to the same event"
    )
    assert ev_by_art[id_c] != ev_by_art[id_a], (
        f"[{test_label}] C must belong to a separate event from A and B"
    )

    memberships = [
        {art.article_id for art in updated if art.event_id == ev.event_id}
        for ev in events
    ]
    assert {id_a, id_b} in memberships, (
        f"[{test_label}] Expected exact cluster {{{id_a}, {id_b}}} in {memberships}"
    )
    assert {id_c} in memberships, (
        f"[{test_label}] Expected exact cluster {{{id_c}}} in {memberships}"
    )


def test_step875_pricing_release_compatibility_unit():
    """STEP 8.7.5: Compatibility unit tests for is_candidate_compatible_with_cluster().

    Verifies:
        - GPT-5 pricing is compatible with GPT-5 release.
        - Claude pricing is NOT compatible with GPT-5 release (similarity 0.0).
        - GPT-5 release is NOT compatible with Claude pricing (similarity 0.0).
        - GPT-5 pricing is NOT compatible with Claude pricing (similarity 0.0).
        - Disjoint model context produces similarity == 0.0 under the compatibility contract.
    """
    rel_art = Article(
        article_id="unit_gpt5_rel",
        title="OpenAI releases GPT-5 frontier AI model",
        url="https://openai.com/gpt5-release",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    prc_art = Article(
        article_id="unit_gpt5_pricing",
        title="OpenAI announces GPT-5 pricing and API subscription tiers",
        url="https://theverge.com/gpt5-pricing",
        source="The Verge",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    cld_art = Article(
        article_id="unit_claude_pricing",
        title="OpenAI announces Claude model API pricing update",
        url="https://techcrunch.com/claude-pricing",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "Claude"],
    )

    articles = [rel_art, prc_art, cld_art]
    features = [extract_clustering_features(a) for a in articles]
    min_cohesion = DEFAULT_SIMILARITY_THRESHOLD * MIN_CLUSTER_COHESION_RATIO

    # 1. GPT-5 pricing candidate (idx 1) compatible with GPT-5 release cluster [0]
    comp, sim = is_candidate_compatible_with_cluster(
        candidate_idx=1,
        cluster=[0],
        articles=articles,
        features=features,
        window_hours=DEFAULT_EVENT_WINDOW_HOURS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        min_cohesion_threshold=min_cohesion,
    )
    assert comp is True, "GPT-5 pricing must be compatible with GPT-5 release"
    assert sim > 0.0, "GPT-5 pricing similarity to GPT-5 release must be positive"

    # 2. Claude pricing candidate (idx 2) NOT compatible with GPT-5 release cluster [0]
    comp, sim = is_candidate_compatible_with_cluster(
        candidate_idx=2,
        cluster=[0],
        articles=articles,
        features=features,
        window_hours=DEFAULT_EVENT_WINDOW_HOURS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        min_cohesion_threshold=min_cohesion,
    )
    assert comp is False, "Claude pricing must NOT be compatible with GPT-5 release"
    assert sim == 0.0, "Disjoint model context must yield similarity 0.0"

    # 3. GPT-5 release candidate (idx 0) NOT compatible with Claude pricing cluster [2]
    comp, sim = is_candidate_compatible_with_cluster(
        candidate_idx=0,
        cluster=[2],
        articles=articles,
        features=features,
        window_hours=DEFAULT_EVENT_WINDOW_HOURS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        min_cohesion_threshold=min_cohesion,
    )
    assert comp is False, "GPT-5 release must NOT be compatible with Claude pricing"
    assert sim == 0.0, "Disjoint model context must yield similarity 0.0"

    # 4. GPT-5 pricing candidate (idx 1) NOT compatible with Claude pricing cluster [2]
    comp, sim = is_candidate_compatible_with_cluster(
        candidate_idx=1,
        cluster=[2],
        articles=articles,
        features=features,
        window_hours=DEFAULT_EVENT_WINDOW_HOURS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        min_cohesion_threshold=min_cohesion,
    )
    assert comp is False, "GPT-5 pricing must NOT be compatible with Claude pricing"
    assert sim == 0.0, "Disjoint model context must yield similarity 0.0"


def test_step874_exhaustive_permutation_clustering():
    """Test STEP 8.7.4 / 8.7.5: Validate that clustering is deterministic across all 6 permutations
    of input articles A, B, and C using freshly created fixtures per iteration.

    Articles:
        A: OpenAI GPT-5 release
        B: GPT-5 pricing announcement
        C: Unrelated Claude pricing update

    Permutations:
        [A, B, C], [A, C, B], [B, A, C], [B, C, A], [C, A, B], [C, B, A]

    Distinguishes between:
        1. Input list order: Tested with fresh unprioritized fixtures.
        2. Internal processing order: Controlled and verified via descending priority_score
           so _sort_article_indices_for_clustering sequences through all 6 orders.
        3. Final event membership: Verified to be {A, B} and {C} across all scenarios.
    """
    def _create_articles():
        a = Article(
            article_id="step874_a_gpt5_rel",
            title="OpenAI releases GPT-5 frontier AI model",
            url="https://openai.com/gpt5-release",
            source="OpenAI",
            category="frontier_models",
            tags=["OpenAI", "GPT-5"],
        )
        b = Article(
            article_id="step874_b_gpt5_pricing",
            title="OpenAI announces GPT-5 pricing and API subscription tiers",
            url="https://theverge.com/gpt5-pricing",
            source="The Verge",
            category="frontier_models",
            tags=["OpenAI", "GPT-5"],
        )
        c = Article(
            article_id="step874_c_claude_pricing",
            title="OpenAI announces Claude model API pricing update",
            url="https://techcrunch.com/claude-pricing",
            source="TechCrunch",
            category="frontier_models",
            tags=["OpenAI", "Claude"],
        )
        return a, b, c

    permutation_keys = [
        ("ABC", ["A", "B", "C"]),
        ("ACB", ["A", "C", "B"]),
        ("BAC", ["B", "A", "C"]),
        ("BCA", ["B", "C", "A"]),
        ("CAB", ["C", "A", "B"]),
        ("CBA", ["C", "B", "A"]),
    ]

    # 1. Test raw list permutations with fresh fixtures per iteration
    for perm_name, keys in permutation_keys:
        a, b, c = _create_articles()
        article_map = {"A": a, "B": b, "C": c}
        perm_articles = [article_map[key] for key in keys]

        events, event_articles, updated = cluster_articles(perm_articles)
        _assert_gpt5_launch_clustering_membership(
            events, updated, f"Raw_{perm_name}", a.article_id, b.article_id, c.article_id
        )

    # 2. Test forced internal processing orders with fresh fixtures per iteration
    # _sort_article_indices_for_clustering sorts by (off, priority_score, published_at, uid) descending.
    # Assigning descending priority_scores strictly forces the internal sequential processing order.
    for perm_name, keys in permutation_keys:
        a, b, c = _create_articles()
        article_map = {"A": a, "B": b, "C": c}
        for rank, key in enumerate(keys):
            article_map[key].priority_score = 100.0 - rank * 10.0

        perm_articles = [article_map[key] for key in keys]

        # Explicitly verify the intended internal processing order
        sorted_indices = _sort_article_indices_for_clustering(perm_articles)
        actual_order = [perm_articles[i].article_id for i in sorted_indices]
        expected_order = [article_map[key].article_id for key in keys]
        assert actual_order == expected_order, (
            f"[{perm_name}] Expected processing order {expected_order}, got {actual_order}"
        )

        events, event_articles, updated = cluster_articles(perm_articles)
        _assert_gpt5_launch_clustering_membership(
            events, updated, f"Forced_{perm_name}", a.article_id, b.article_id, c.article_id
        )


def test_step874_exact_chaining_scenarios():
    """Test STEP 8.7.4 / 8.7.5: Exact chaining scenarios with isolated fresh fixtures:
    1. GPT-5 release first, then GPT-5 pricing, then unrelated Claude pricing.
    2. GPT-5 pricing first, then GPT-5 release, then unrelated Claude pricing.
    3. Unrelated Claude pricing first, then GPT-5 release, then GPT-5 pricing.
    4. GPT-5 release and unrelated Claude pricing first, then GPT-5 pricing.
    5. GPT-5 pricing and unrelated Claude pricing first, then GPT-5 release.

    Verify final event membership remains:
        {GPT-5 release, GPT-5 pricing}
        {Claude pricing}
    """
    def _create_fresh():
        return (
            Article(
                article_id="step874_rel",
                title="OpenAI releases GPT-5 frontier AI model",
                url="https://openai.com/gpt5-release",
                source="OpenAI",
                category="frontier_models",
                tags=["OpenAI", "GPT-5"],
            ),
            Article(
                article_id="step874_pricing",
                title="OpenAI announces GPT-5 pricing and API subscription tiers",
                url="https://theverge.com/gpt5-pricing",
                source="The Verge",
                category="frontier_models",
                tags=["OpenAI", "GPT-5"],
            ),
            Article(
                article_id="step874_claude_pricing",
                title="OpenAI announces Claude model API pricing update",
                url="https://techcrunch.com/claude-pricing",
                source="TechCrunch",
                category="frontier_models",
                tags=["OpenAI", "Claude"],
            ),
        )

    # Scenarios 1 to 5 verified through end-to-end clustering with guaranteed sequence
    chain_scenarios = [
        ("Scenario 1: rel -> pricing -> claude_pricing", ["step874_rel", "step874_pricing", "step874_claude_pricing"]),
        ("Scenario 2: pricing -> rel -> claude_pricing", ["step874_pricing", "step874_rel", "step874_claude_pricing"]),
        ("Scenario 3: claude_pricing -> rel -> pricing", ["step874_claude_pricing", "step874_rel", "step874_pricing"]),
        ("Scenario 4: rel & claude_pricing first -> pricing", ["step874_rel", "step874_claude_pricing", "step874_pricing"]),
        ("Scenario 5: pricing & claude_pricing first -> rel", ["step874_pricing", "step874_claude_pricing", "step874_rel"]),
    ]

    for sc_name, order_keys in chain_scenarios:
        art_rel, art_prc, art_cld = _create_fresh()
        art_map = {
            "step874_rel": art_rel,
            "step874_pricing": art_prc,
            "step874_claude_pricing": art_cld,
        }
        for rank, key in enumerate(order_keys):
            art_map[key].priority_score = 100.0 - rank * 10.0

        articles_to_cluster = [art_map[k] for k in order_keys]
        events, _, updated = cluster_articles(articles_to_cluster)
        _assert_gpt5_launch_clustering_membership(
            events, updated, sc_name, "step874_rel", "step874_pricing", "step874_claude_pricing"
        )

    # In-depth compatibility verification for Scenarios 4 and 5 with fresh fixtures
    art_rel, art_prc, art_cld = _create_fresh()
    articles = [art_rel, art_prc, art_cld]  # idx 0: rel, 1: pricing, 2: claude_pricing
    features = [extract_clustering_features(a) for a in articles]
    min_cohesion = DEFAULT_SIMILARITY_THRESHOLD * MIN_CLUSTER_COHESION_RATIO

    # Scenario 4 detailed check: clusters [rel] and [claude_pricing], candidate pricing (idx 1)
    comp_to_rel, sim_to_rel = is_candidate_compatible_with_cluster(
        candidate_idx=1,
        cluster=[0],
        articles=articles,
        features=features,
        window_hours=DEFAULT_EVENT_WINDOW_HOURS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        min_cohesion_threshold=min_cohesion,
    )
    comp_to_claude, sim_to_claude = is_candidate_compatible_with_cluster(
        candidate_idx=1,
        cluster=[2],
        articles=articles,
        features=features,
        window_hours=DEFAULT_EVENT_WINDOW_HOURS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        min_cohesion_threshold=min_cohesion,
    )
    assert comp_to_rel is True, "Scenario 4: pricing candidate must be compatible with GPT-5 release cluster"
    assert sim_to_rel > 0.0, "Scenario 4: similarity must be positive"
    assert comp_to_claude is False, "Scenario 4: pricing candidate must NOT be compatible with Claude pricing cluster"
    assert sim_to_claude == 0.0, "Scenario 4: similarity must be 0.0 due to disjoint models"

    # Scenario 5 detailed check: clusters [pricing] and [claude_pricing], candidate release (idx 0)
    comp_to_prc, sim_to_prc = is_candidate_compatible_with_cluster(
        candidate_idx=0,
        cluster=[1],
        articles=articles,
        features=features,
        window_hours=DEFAULT_EVENT_WINDOW_HOURS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        min_cohesion_threshold=min_cohesion,
    )
    comp_rel_to_claude, sim_rel_to_claude = is_candidate_compatible_with_cluster(
        candidate_idx=0,
        cluster=[2],
        articles=articles,
        features=features,
        window_hours=DEFAULT_EVENT_WINDOW_HOURS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        min_cohesion_threshold=min_cohesion,
    )
    assert comp_to_prc is True, "Scenario 5: GPT-5 release candidate must be compatible with GPT-5 pricing cluster"
    assert sim_to_prc > 0.0, "Scenario 5: similarity must be positive"
    assert comp_rel_to_claude is False, "Scenario 5: GPT-5 release candidate must NOT be compatible with Claude pricing cluster"
    assert sim_rel_to_claude == 0.0, "Scenario 5: similarity must be 0.0"











