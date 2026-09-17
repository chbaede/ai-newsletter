from ai_newsletter.clustering import cluster_articles, select_primary_article
from ai_newsletter.models import Article


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
    """1. Same company, different events: GPT-5 launch vs GPT-5 pricing vs London office."""
    a1 = Article(
        title="OpenAI announces GPT-5",
        url="https://openai.com/gpt-5",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI"],
    )
    a2 = Article(
        title="OpenAI announces GPT-5 pricing and subscription tiers",
        url="https://theverge.com/gpt-5-pricing",
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
    """2. Same model, different actions: Release vs Pricing vs Security vulnerability."""
    a1 = Article(
        title="OpenAI releases GPT-5 flagship model",
        url="https://openai.com/gpt-5-release",
        source="OpenAI",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
    )
    a2 = Article(
        title="OpenAI changes GPT-5 pricing for API developer accounts",
        url="https://techcrunch.com/gpt5-price-change",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
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
    """Bridge Test C: Launch story (A) vs Pricing announcement (B) must remain cleanly separated."""
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
        title="OpenAI introduces new subscription pricing and API rates for GPT-5",
        url="https://techcrunch.com/openai-gpt-5-pricing",
        source="TechCrunch",
        category="frontier_models",
        tags=["OpenAI", "GPT-5"],
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




