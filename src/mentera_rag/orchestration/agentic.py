"""
Agentic RAG State Graph Implementation — Mentera RAG Pipeline.

Uses LangGraph state machine to implement self-correcting retrieval loops.
Applies query rewriting to optimize semantic search recall and passes tenant
isolation filters through the retrieval nodes.

Returns retrieved and graded contexts without final LLM answer generation.
"""

from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from mentera_rag.chunking.schemas import Chunk
from mentera_rag.generation.base import BaseLLMProvider
from mentera_rag.generation.prompts import (
    QUERY_REWRITER_SYSTEM_PROMPT,
    QUERY_REWRITER_USER_PROMPT,
    RELEVANCE_GRADER_SYSTEM_PROMPT,
    RELEVANCE_GRADER_USER_PROMPT,
)
from mentera_rag.retrieval.base import BaseRetriever


class RAGState(TypedDict):
    """LangGraph State carrying metadata through agent execution graph."""

    query: str
    original_query: str
    documents: list[Chunk]
    generation: str
    retry_count: int
    is_relevant: bool
    filters: dict[str, Any] | None


class AgenticRAGGraph:
    """
    Agentic RAG state machine for context retrieval with query rewriting and self-correction.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_provider: BaseLLMProvider,
        max_retries: int = 2,
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.max_retries = max_retries
        self.workflow = self._build_graph()

    def _retrieve_node(self, state: RAGState) -> dict[str, Any]:
        """Graph Node: Retrieve documents for current query with tenant filters."""
        chunks = self.retriever.retrieve(
            state["query"],
            top_k=5,
            filters=state.get("filters"),
        )
        return {"documents": chunks}

    def _grade_documents_node(self, state: RAGState) -> dict[str, Any]:
        """
        Graph Node: Grade retrieved document passages for semantic relevance using LLM Evaluator.
        """
        documents = state.get("documents", [])
        if not documents:
            return {"is_relevant": False}

        query = state["query"]
        relevant_found = False

        # Grade top retrieved passages using LLM Evaluator
        for doc in documents[:3]:
            try:
                prompt = RELEVANCE_GRADER_USER_PROMPT.format(document=doc.text[:1000], query=query)
                res = self.llm_provider.generate(
                    prompt=prompt, system_prompt=RELEVANCE_GRADER_SYSTEM_PROMPT
                )
                res_str = res.strip().lower()
                if "true" in res_str or '"relevant": true' in res_str or "yes" in res_str:
                    relevant_found = True
                    break
            except Exception:
                # Deterministic term overlap fallback (filtering stop words)
                stop_words = {
                    "a",
                    "an",
                    "the",
                    "is",
                    "are",
                    "was",
                    "were",
                    "for",
                    "and",
                    "or",
                    "in",
                    "on",
                    "at",
                    "to",
                    "of",
                    "what",
                    "with",
                    "do",
                    "does",
                    "by",
                    "i",
                    "you",
                    "it",
                    "this",
                    "that",
                    "how",
                    "can",
                }
                query_words = set(query.lower().split()) - stop_words
                doc_words = set(doc.text.lower().split()) - stop_words
                if len(query_words.intersection(doc_words)) > 0:
                    relevant_found = True
                    break

        return {"is_relevant": relevant_found}

    def _rewrite_query_node(self, state: RAGState) -> dict[str, Any]:
        """Graph Node: Rewrite query to increase search recall."""
        prompt = QUERY_REWRITER_USER_PROMPT.format(query=state["query"])
        new_query = self.llm_provider.generate(
            prompt=prompt, system_prompt=QUERY_REWRITER_SYSTEM_PROMPT
        )
        return {
            "query": new_query or state["query"],
            "retry_count": state["retry_count"] + 1,
        }

    def _decide_to_finish(self, state: RAGState) -> str:
        """Graph Edge Decision: Decide whether to finish or rewrite query."""
        if state["is_relevant"] or state["retry_count"] >= self.max_retries:
            return "finish"
        return "rewrite_query"

    def _build_graph(self) -> Any:
        """Construct LangGraph StateGraph."""
        builder = StateGraph(RAGState)

        # Add Nodes
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade_documents", self._grade_documents_node)
        builder.add_node("rewrite_query", self._rewrite_query_node)

        # Add Edges
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "grade_documents")
        builder.add_conditional_edges(
            "grade_documents",
            self._decide_to_finish,
            {
                "finish": END,
                "rewrite_query": "rewrite_query",
            },
        )
        builder.add_edge("rewrite_query", "retrieve")

        return builder.compile()

    def run(self, query: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute Agentic RAG Graph."""
        initial_state: RAGState = {
            "query": query,
            "original_query": query,
            "documents": [],
            "generation": "",
            "retry_count": 0,
            "is_relevant": False,
            "filters": filters,
        }
        final_state = self.workflow.invoke(initial_state)

        return {
            "query": query,
            "final_query": final_state.get("query"),
            "answer": "",  # Context-only pipeline
            "retrieved_chunks": final_state.get("documents", []),
            "retry_count": final_state.get("retry_count", 0),
            "pipeline_type": "agentic",
        }
