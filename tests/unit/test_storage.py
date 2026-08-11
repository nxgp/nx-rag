import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from mentera_rag.storage.local import LocalStorageStore
from mentera_rag.storage.factory import StorageFactory
from mentera_rag.storage.s3 import S3StorageStore
from mentera_rag.storage.azure_blob import AzureBlobStorageStore
from mentera_rag.storage.gcs import GCSStorageStore


@pytest.fixture
def temp_storage_dir(tmp_path) -> Path:
    return tmp_path / "local_store"


@pytest.mark.unit
def test_local_storage_basic(temp_storage_dir):
    """Test LocalStorageStore basic CRUD operations."""
    store = LocalStorageStore(root_dir=temp_storage_dir)

    key = "tenant1/provider2/file1.txt"
    content = b"Local file storage contents."

    # Test file upload
    store.upload_file(key, content)

    # Test exists
    assert store.exists(key) is True

    # Test download
    downloaded = store.download(key)
    assert downloaded == content

    # Test delete
    store.delete(key)
    assert store.exists(key) is False


@pytest.mark.unit
def test_local_storage_path_traversal(temp_storage_dir):
    """Test directory traversal prevention in LocalStorageStore."""
    store = LocalStorageStore(root_dir=temp_storage_dir)

    # Traversal key attempting to read parent files
    traversal_key = "../../../etc/passwd"

    with pytest.raises(ValueError, match="Path traversal detected"):
        store.download(traversal_key)

    with pytest.raises(ValueError, match="Path traversal detected"):
        store.upload_file(traversal_key, b"hack")


@pytest.mark.unit
@patch("os.getenv")
def test_storage_factory_local(mock_getenv, temp_storage_dir):
    """Test factory resolution of LocalStorageStore."""
    mock_getenv.return_value = str(temp_storage_dir)
    store = StorageFactory.get_store(provider_type="local")
    assert isinstance(store, LocalStorageStore)
    assert store.root_dir.resolve() == temp_storage_dir.resolve()


@pytest.mark.unit
@patch("mentera_rag.storage.s3.boto3.client")
def test_storage_factory_s3(mock_boto):
    """Test factory resolution of S3StorageStore."""
    store = StorageFactory.get_store(provider_type="s3")
    assert isinstance(store, S3StorageStore)


@pytest.mark.unit
@patch("os.getenv")
def test_storage_factory_azure_missing_conn_string(mock_getenv):
    """Test that Azure Blob store raises error if connection string env variable is missing."""
    mock_getenv.return_value = ""
    with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
        StorageFactory.get_store(provider_type="azure_blob")


@pytest.mark.unit
@patch("os.getenv")
def test_storage_factory_azure_success(mock_getenv):
    """Test factory resolution of AzureBlobStorageStore."""
    mock_getenv.return_value = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net"
    # Patch service client property since connection string is dummy
    with patch.object(AzureBlobStorageStore, "service_client", MagicMock()):
        store = StorageFactory.get_store(provider_type="azure_blob")
        assert isinstance(store, AzureBlobStorageStore)


@pytest.mark.unit
@patch("google.cloud.storage.Client")
def test_storage_factory_gcs(mock_gcs_client):
    """Test factory resolution of GCSStorageStore."""
    store = StorageFactory.get_store(provider_type="gcs")
    assert isinstance(store, GCSStorageStore)
