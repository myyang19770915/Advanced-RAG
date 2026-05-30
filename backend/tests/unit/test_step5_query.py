from app.pipelines import step5_query
from app.pipelines.step4_ingestion import embed_and_upsert
from app.providers.embedding import MockEmbedding
from app.providers.llm import MockLLM
from app.providers.vector_store import InMemoryVectorStore
from app.schemas import ChunkPayload


async def test_full_query_flow():
    llm = MockLLM()
    emb = MockEmbedding(dim=32)
    vs = InMemoryVectorStore()

    chunks = [
        ChunkPayload(
            text="Working hours are limited to 40 per week.",
            original_file="labour.pdf",
            source_file="labour.md",
            chunk_index=0,
            project_name="labour",
        ),
        ChunkPayload(
            text="Overtime pay is 1.5x regular wage.",
            original_file="labour.pdf",
            source_file="labour.md",
            chunk_index=1,
            project_name="labour",
        ),
    ]
    n = await embed_and_upsert(
        chunks, collection="rag_labour", embedding=emb, vector_store=vs
    )
    assert n == 2

    answer, hits = await step5_query.run_query(
        question="What is overtime pay?",
        collection="rag_labour",
        project_name="labour",
        llm=llm,
        embedding=emb,
        vector_store=vs,
        top_k=2,
    )
    assert answer  # Mock returns a stub answer string
    assert len(hits) == 2
    assert all(h.payload["project_name"] == "labour" for h in hits)
