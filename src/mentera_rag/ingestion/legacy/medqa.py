"""
MedQA Dataset Loader & Normalizer.

MedQA contains USMLE-style multiple-choice questions. Per ADR 0012, questions
do NOT carry passage-level relevance annotations, so qrels list is empty.
"""

import hashlib
from typing import Any

from datasets import load_dataset

from mentera_rag.ingestion.base import BaseDatasetLoader
from mentera_rag.ingestion.schemas import Document, Qrel, Query


class MedQALoader(BaseDatasetLoader):
    """
    Loads MedQA dataset (truehealth/medqa) and builds MCQ query set.
    """

    def __init__(self, dataset_name: str = "truehealth/medqa"):
        self.dataset_name = dataset_name

    def fetch_raw(self) -> Any:
        """Fetch raw records using HuggingFace datasets."""
        return load_dataset(self.dataset_name, split="test")  # nosec B615

    def _hash_text(self, text: str) -> str:
        """Create a deterministic SHA-256 hash prefix for text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def process(self) -> tuple[list[Document], list[Query], list[Qrel]]:
        raw_data = self.fetch_raw()

        documents: list[Document] = []
        queries: list[Query] = []
        qrels: list[Qrel] = []

        for idx, item in enumerate(raw_data):
            question_text = item["question"]
            options_dict = item.get("options", {})
            gold_answer = item.get("answer", "")

            query_id = f"medqa_q_{idx}_{self._hash_text(question_text)}"
            queries.append(
                Query(
                    query_id=query_id,
                    text=question_text,
                    metadata={
                        "options": options_dict,
                        "gold_answer": gold_answer,
                        "source": "medqa",
                        "meta_info": item.get("meta_info", ""),
                        "answer_idx": item.get("answer_idx", ""),
                    },
                )
            )
        return documents, queries, qrels
