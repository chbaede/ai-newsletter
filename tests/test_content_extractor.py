from ai_newsletter.content_extractor import extract_text_from_html, extract_usable_article_text


def test_extract_text_from_html():
    html_doc = """
    <html>
      <head><title>Test</title></head>
      <body>
        <nav><a href="/">Home</a></nav>
        <article>
          <h1>OpenAI Announces GPT-4o</h1>
          <p>OpenAI has officially launched GPT-4o, its newest flagship model that reasons across audio, vision, and text in real time.</p>
          <p>The model will be made available for both free and paid ChatGPT tiers over the coming weeks.</p>
        </article>
        <footer><p>All rights reserved. Cookie policy.</p></footer>
      </body>
    </html>
    """
    text = extract_text_from_html(html_doc)
    assert "OpenAI has officially launched GPT-4o" in text
    assert "Cookie policy" not in text
    assert "Home" not in text


def test_extract_usable_article_text():
    raw_entry = {
        "content": [{"value": "<p>DeepMind researchers published a breakthrough on AlphaFold 3 capabilities in biological structure prediction.</p>"}],
        "summary": "AlphaFold 3 breakthrough announced.",
    }
    extracted = extract_usable_article_text(
        raw_entry=raw_entry,
        url="https://example.com/alphafold",
        title="AlphaFold 3",
    )
    assert extracted.source_type == "rss_content"
    assert "DeepMind researchers published" in extracted.text

