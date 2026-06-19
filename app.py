from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from config import load_env, get_llm
from ingestion import build_retriever
from tools import web_search_fallback


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader. Given a retrieved document and a user question, "
     "respond ONLY with 'yes' if the document is relevant to the question, "
     "or 'no' if it is not. No explanation."),
    ("human", "Document:\n{document}\n\nQuestion: {question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query optimizer. Rewrite the user question to be more specific "
     "and suitable for a web search engine. Return ONLY the rewritten query."),
    ("human", "Original question: {question}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful customer support assistant. "
     "Answer the question using ONLY the provided context. "
     "If the context doesn't contain enough information, say so clearly."),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}"),
])


def retrieve(state: GraphState, retriever) -> dict:
    print("\n[NODE] retrieve")
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def grade_documents(state: GraphState, llm) -> dict:
    print("[NODE] grade_documents")
    grader_chain = GRADE_PROMPT | llm | StrOutputParser()
    filtered = []
    for doc in state["documents"]:
        score = grader_chain.invoke({
            "document": doc.page_content,
            "question": state["question"],
        }).strip().lower()
        if score == "yes":
            filtered.append(doc)
    print(f"  {len(filtered)}/{len(state['documents'])} docs passed grading")
    return {"documents": filtered}


def transform_query(state: GraphState, llm) -> dict:
    print("[NODE] transform_query")
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    new_question = rewrite_chain.invoke({"question": state["question"]}).strip()
    print(f"  Rewritten: '{new_question}'")
    return {"question": new_question}


def web_search(state: GraphState) -> dict:
    print("[NODE] web_search")
    docs = web_search_fallback(state["question"])
    return {"documents": docs}


def generate(state: GraphState, llm) -> dict:
    print("[NODE] generate")
    context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
    gen_chain = GENERATE_PROMPT | llm | StrOutputParser()
    answer = gen_chain.invoke({
        "context": context,
        "question": state["question"],
    })
    return {"generation": answer}


def route_after_grading(state: GraphState) -> str:
    if state["documents"]:
        print("[ROUTE] -> generate (relevant docs found)")
        return "generate"
    print("[ROUTE] -> transform_query (no relevant docs)")
    return "transform_query"


def build_graph(retriever):
    llm = get_llm()

    def _retrieve(state):       return retrieve(state, retriever)
    def _grade(state):          return grade_documents(state, llm)
    def _transform(state):      return transform_query(state, llm)
    def _web_search(state):     return web_search(state)
    def _generate(state):       return generate(state, llm)

    graph = StateGraph(GraphState)

    graph.add_node("retrieve",        _retrieve)
    graph.add_node("grade_documents", _grade)
    graph.add_node("transform_query", _transform)
    graph.add_node("web_search",      _web_search)
    graph.add_node("generate",        _generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",        "grade_documents")
    graph.add_edge("transform_query", "web_search")
    graph.add_edge("web_search",      "generate")
    graph.add_edge("generate",        END)

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate":        "generate",
            "transform_query": "transform_query",
        },
    )

    return graph.compile()
