"""
PubMedQA Dataset Loader & Normalizer.

PubMedQA consists of biomedical abstracts with explicit question-context pairings.
Context passages serve as gold relevance labels (qrels) for IR metrics.
"""

import hashlib
from typing import Any

from datasets import load_dataset

from mentera_rag.ingestion.base import BaseDatasetLoader
from mentera_rag.ingestion.schemas import Document, Qrel, Query


class PubMedQALoader(BaseDatasetLoader):
    """
    Loads PubMedQA dataset (qiaojin/PubMedQA, pqa_labeled subset) and builds corpus + qrels.
    """

    def __init__(self, dataset_name: str = "qiaojin/PubMedQA", subset: str | None = "pqa_labeled"):
        self.dataset_name = dataset_name
        self.subset = subset

    def fetch_raw(self) -> Any:
        """Fetch raw records via HuggingFace datasets."""
        if self.subset:
            return load_dataset(self.dataset_name, self.subset, split="train")  # nosec B615
        return load_dataset(self.dataset_name, split="train")  # nosec B615

    def _generate_doc_id(self, pmid: str, idx: int) -> str:
        """Generate a deterministic document ID from PMID and passage index."""
        return f"pubmed_{pmid}_ctx_{idx}"

    def _generate_query_id(self, pmid: str) -> str:
        """Generate a deterministic query ID from PMID."""
        return f"pubmed_q_{pmid}"

    def process(self) -> tuple[list[Document], list[Query], list[Qrel]]:
        raw_data = self.fetch_raw()

        documents: list[Document] = []
        queries: list[Query] = []
        qrels: list[Qrel] = []

        seen_doc_ids = set()

        for item in raw_data:
            question_text = item["question"]
            pmid = str(item.get("pubid", hashlib.sha256(question_text.encode()).hexdigest()[:10]))
            long_answer = item.get("long_answer", "")

            contexts = item.get("context", {})
            if isinstance(contexts, dict):
                raw_passages = contexts.get("contexts") or contexts.get("passages") or []
                passages = raw_passages if isinstance(raw_passages, list) else [str(raw_passages)]
                labels = contexts.get("labels") or []
                meshes = contexts.get("meshes") or []
            elif isinstance(contexts, list):
                passages = contexts
                labels = []
                meshes = item.get("meshes") or []
            else:
                passages = [str(contexts)] if contexts else []
                labels = []
                meshes = []

            query_id = self._generate_query_id(pmid)

            queries.append(
                Query(
                    id=query_id,
                    text=question_text,
                    gold_answer=long_answer,
                    source="pubmedqa",
                    metadata={
                        "pmid": pmid,
                        "final_decision": item.get("final_decision", ""),
                        "meshes": meshes,
                    },
                )
            )

            for idx, passage in enumerate(passages):
                doc_id = self._generate_doc_id(pmid, idx)
                section_label = labels[idx] if idx < len(labels) else "ABSTRACT"

                if doc_id not in seen_doc_ids:
                    documents.append(
                        Document(
                            id=doc_id,
                            title=f"PubMed Abstract {pmid} - Section {section_label}",
                            text=passage,
                            source="pubmedqa",
                            metadata={
                                "pmid": pmid,
                                "section_label": section_label,
                                "meshes": meshes,
                            },
                        )
                    )
                    seen_doc_ids.add(doc_id)

                qrels.append(Qrel(query_id=query_id, document_id=doc_id, relevance=1))

        return documents, queries, qrels
