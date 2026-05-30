from app.providers.embedding import MockEmbedding
from app.providers.vector_store import InMemoryVectorStore, VectorPoint


async def test_inmemory_upsert_and_search():
    store = InMemoryVectorStore()
    emb = MockEmbedding(dim=16)
    await store.ensure_collection("c1", emb.dim)
    texts = ["the cat sat on the mat", "deep learning models", "machine learning intro"]
    vecs = await emb.embed(texts)
    points = [
        VectorPoint(id=str(i), vector=v, payload={"text": t, "project_name": "p"})
        for i, (t, v) in enumerate(zip(texts, vecs))
    ]
    assert await store.upsert("c1", points) == 3

    query_vec = (await emb.embed(["machine learning intro"]))[0]
    hits = await store.search("c1", query_vec, top_k=2)
    assert len(hits) == 2
    # The exact-match input should be the best hit
    assert hits[0].payload["text"] == "machine learning intro"


async def test_inmemory_filter():
    store = InMemoryVectorStore()
    emb = MockEmbedding(dim=8)
    vecs = await emb.embed(["a", "b"])
    await store.ensure_collection("c", emb.dim)
    await store.upsert(
        "c",
        [
            VectorPoint(id="1", vector=vecs[0], payload={"project_name": "p1"}),
            VectorPoint(id="2", vector=vecs[1], payload={"project_name": "p2"}),
        ],
    )
    hits = await store.search("c", vecs[0], top_k=5, filters={"project_name": "p2"})
    assert len(hits) == 1 and hits[0].id == "2"
    assert await store.count("c", filters={"project_name": "p1"}) == 1
