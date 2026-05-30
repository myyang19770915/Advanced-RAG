async def test_settings_returns_provider_info(client):
    r = await client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    assert data["llm_provider"] in {"mock", "openai"}
    assert "embedding_provider" in data
