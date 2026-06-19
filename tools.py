from ddgs import DDGS
from langchain_core.documents import Document


def web_search_fallback(query: str) -> list[Document]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    if not results:
        return [Document(
            page_content="No web results found.",
            metadata={"source": "web_search", "query": query},
        )]

    combined = "\n\n".join(
        f"{r.get('title', '')}\n{r.get('body', '')}" for r in results
    )
    return [Document(
        page_content=combined,
        metadata={"source": "web_search", "query": query},
    )]
