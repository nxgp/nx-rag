"""
Dataset Artifact Versioning & Manifest Generation.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from mentera_rag.ingestion.schemas import DatasetManifest, Document, Qrel, Query


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a target file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_jsonl(filepath: Path, items: list[Any]) -> None:
    """Writes a list of Pydantic models or dicts to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in items:
            data = item.model_dump() if hasattr(item, "model_dump") else item
            f.write(json.dumps(data) + "\n")


def create_manifest(
    output_dir: Path,
    dataset_name: str,
    version: str,
    docs: list[Document],
    queries: list[Query],
    qrels: list[Qrel],
) -> DatasetManifest:
    """
    Serialized ingestion outputs to JSONL and generates an audit manifest.
    """

    docs_file = output_dir / f"{dataset_name}_documents.jsonl"
    queries_file = output_dir / f"{dataset_name}_queries.jsonl"
    qrels_file = output_dir / f"{dataset_name}_qrels.jsonl"

    # Write JSONL artifacts
    write_jsonl(docs_file, docs)
    write_jsonl(queries_file, queries)
    write_jsonl(qrels_file, qrels)

    # Compute checksums
    checksums = {
        docs_file.name: compute_file_sha256(docs_file),
        queries_file.name: compute_file_sha256(queries_file),
        qrels_file.name: compute_file_sha256(qrels_file),
    }

    manifest = DatasetManifest(
        dataset_name=dataset_name,
        version=version,
        document_count=len(docs),
        query_count=len(queries),
        qrel_count=len(qrels),
        sha256_checksums=checksums,
    )

    manifest_path = output_dir / f"{dataset_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    return manifest
