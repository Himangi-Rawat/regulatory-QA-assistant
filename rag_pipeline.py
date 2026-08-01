"""
rag_pipeline.py
----------------
Takes a user's question, retrieves the most relevant chunks from the
vector store, and asks Gemini to answer USING ONLY those chunks — with
a mandatory citation for every claim.
"""

from vector_store import VectorStore, get_client

GENERATION_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = """You are a regulatory compliance assistant. You answer questions \
strictly using the CONTEXT provided below, which consists of excerpts from official \
regulatory/compliance PDF documents.

Rules you must follow:
1. Only use facts that appear in the CONTEXT. Do not use outside knowledge.
2. For every factual claim you make, cite it immediately in this exact format: \
[Source: <document name>, Page <page number>].
3. If the CONTEXT does not contain enough information to answer the question, \
say clearly: "I could not find this in the provided documents." Do not guess \
or make up a citation.
4. Keep answers concise and clause-focused — this is for engineers who need a \
fast, trustworthy answer, not a long essay.
"""


def build_context_block(results: list[tuple]) -> str:
    """Format retrieved chunks into a labeled context block the model can cite from."""
    blocks = []
    for chunk, distance in results:
        blocks.append(
            f"[Document: {chunk.doc_name} | Page: {chunk.page_number}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(store: VectorStore, question: str, top_k: int = 5) -> dict:
    """
    Full RAG call: retrieve top_k chunks, build a citation-safe prompt,
    call Gemini, and return both the answer and the raw sources used.
    """
    results = store.search(question, top_k=top_k)

    if not results:
        return {
            "answer": "The document index is empty — please upload and process a PDF first.",
            "sources": [],
        }

    context_block = build_context_block(results)
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_block}\n\nQUESTION: {question}\n\nANSWER:"

    client = get_client()
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    sources = [
        {"doc_name": c.doc_name, "page_number": c.page_number, "snippet": c.text, "distance": d}
        for c, d in results
    ]

    return {"answer": response.text, "sources": sources}