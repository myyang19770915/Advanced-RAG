from app.providers.embedding import MockEmbedding


async def test_mock_embedding_deterministic():
    emb = MockEmbedding(dim=16)
    v1 = await emb.embed(["hello", "world"])
    v2 = await emb.embed(["hello", "world"])
    assert v1 == v2
    assert len(v1[0]) == 16
    # L2 normalised
    assert abs(sum(x * x for x in v1[0]) - 1.0) < 1e-6


async def test_mock_embedding_different_inputs_differ():
    emb = MockEmbedding(dim=16)
    a, b = await emb.embed(["alpha", "beta"])
    assert a != b
