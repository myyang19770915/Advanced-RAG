async def test_create_and_list_project(client):
    resp = await client.post("/api/projects", json={"name": "demo", "description": "x"})
    assert resp.status_code == 201, resp.text
    proj = resp.json()
    assert proj["name"] == "demo"
    assert proj["collection_name"].startswith("rag_")

    resp = await client.get("/api/projects")
    assert resp.status_code == 200
    assert any(p["name"] == "demo" for p in resp.json())


async def test_duplicate_project_name_conflict(client):
    await client.post("/api/projects", json={"name": "dup"})
    resp = await client.post("/api/projects", json={"name": "dup"})
    assert resp.status_code == 409


async def test_get_missing_project_404(client):
    resp = await client.get("/api/projects/nonexistent")
    assert resp.status_code == 404
