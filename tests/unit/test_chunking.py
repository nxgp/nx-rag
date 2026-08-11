import pytest

from mentera_rag.chunking.factory import ChunkerFactory
from mentera_rag.chunking.recursive import RecursiveCharacterChunker
from mentera_rag.chunking.schemas import Chunk, Document


@pytest.fixture
def sample_document() -> Document:
    """Fixture providing a realistic medical document for chunking tests."""
    content = (
        "Patient presents with acute chest pain and shortness of breath. "
        "Electrocardiogram (ECG) reveals ST-segment elevation in leads II, III, and aVF. "
        "Laboratory findings indicate elevated Troponin I levels at 4.2 ng/mL.\n\n"
        "Diagnosis: Acute Inferior Myocardial Infarction.\n\n"
        "Treatment Plan:\n"
        "1. Immediate administration of Aspirin 325 mg and Clopidogrel 600 mg.\n"
        "2. Transfer to Cardiac Catheterization Lab for emergency PCI.\n"
        "3. Initiate intravenous unfractionated heparin protocol."
    )
    return Document(
        doc_id="doc_med_001",
        content=content,
        source="PubMedQA",
        tenant_id="test_tenant",
        provider_id="test_provider",
        metadata={"author": "Dr. Smith", "accession_id": "ACC9981"},
    )


@pytest.mark.unit
def test_recursive_chunker_basic_split(sample_document: Document):
    """Test that RecursiveCharacterChunker splits content into chunks within target size."""
    chunker = RecursiveCharacterChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk(sample_document)

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, Chunk)
        assert chunk.doc_id == sample_document.doc_id
        assert chunk.chunk_id == f"doc_med_001_c{i}"
        assert chunk.chunk_index == i
        assert len(chunk.text) <= 300
        assert chunk.metadata["source"] == "PubMedQA"
        assert chunk.metadata["author"] == "Dr. Smith"


@pytest.mark.unit
def test_recursive_chunker_short_document():
    """Test that a document shorter than chunk_size produces exactly one chunk."""
    doc = Document(
        doc_id="short_001",
        content="Short clinical note.",
        source="MedQA",
        tenant_id="test_tenant",
        provider_id="test_provider",
    )
    chunker = RecursiveCharacterChunker(chunk_size=500)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Short clinical note."
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "short_001_c0"


@pytest.mark.unit
def test_recursive_chunker_empty_document():
    """Test that an empty document returns an empty list of chunks."""
    doc = Document(
        doc_id="empty_001",
        content="   ",
        source="MedQA",
        tenant_id="test_tenant",
        provider_id="test_provider",
    )
    chunker = RecursiveCharacterChunker(chunk_size=500)
    chunks = chunker.chunk(doc)

    assert chunks == []


@pytest.mark.unit
def test_recursive_chunker_invalid_overlap():
    """Test that initializing overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="strictly smaller than chunk_size"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)


@pytest.mark.unit
def test_chunker_factory_get_chunker():
    """Test instantiating a chunker via ChunkerFactory."""
    chunker = ChunkerFactory.get_chunker("recursive", chunk_size=300, chunk_overlap=20)
    assert isinstance(chunker, RecursiveCharacterChunker)
    assert chunker.chunk_size == 300
    assert chunker.chunk_overlap == 20


@pytest.mark.unit
def test_chunker_factory_invalid_strategy():
    """Test that requesting an unregistered strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown Chunking strategy"):
        ChunkerFactory.get_chunker("unregistered_strategy_name")
