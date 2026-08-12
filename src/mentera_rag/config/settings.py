from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration for the Mentera RAG Pipeline.

    Reads from environment variables and `.env` file automatically.
    Supports multi-cloud embedding providers (AWS Bedrock, Azure OpenAI, GCP Vertex AI)
    and pluggable cloud object storage backends.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    ENV: str = Field(
        default="development",
        description="Execution environment (development, staging, production)",
    )
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # -------------------------------------------------------------------------
    # Qdrant Vector Store
    # -------------------------------------------------------------------------
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant REST API URL")
    QDRANT_API_KEY: str | None = Field(default=None, description="Qdrant API Key (cloud)")
    QDRANT_COLLECTION_NAME: str = Field(
        default="mentera_chunks",
        description="Shared Qdrant collection name (tenant isolation via payload filtering)",
    )

    # -------------------------------------------------------------------------
    # MLflow Experiment Tracking
    # -------------------------------------------------------------------------
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000", description="MLflow tracking server URI"
    )

    # -------------------------------------------------------------------------
    # Embedding Providers
    # -------------------------------------------------------------------------
    DEFAULT_EMBEDDING_PROVIDER: str = Field(
        default="bedrock",
        description="Default embedding provider: 'bedrock', 'azure', or 'gcp'",
    )
    DEFAULT_EMBEDDING_MODEL: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Default embedding model ID (provider-specific)",
    )
    DEFAULT_EMBEDDING_DIMENSION: int = Field(
        default=1024, description="Default embedding vector dimension"
    )

    # AWS Bedrock
    AWS_REGION: str = Field(default="us-east-1", description="AWS Region for Bedrock calls")
    BEDROCK_LLM_MODEL_ID: str = Field(
        default="us.anthropic.claude-3-haiku-20240307-v1:0",
        description="Default Bedrock LLM model ID for query rewriting and agentic RAG",
    )

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint URL (e.g. https://<resource>.openai.azure.com/)",
    )
    AZURE_OPENAI_API_KEY: str | None = Field(default=None, description="Azure OpenAI API Key")
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2024-02-01", description="Azure OpenAI API version"
    )
    AZURE_EMBEDDING_DEPLOYMENT: str = Field(
        default="text-embedding-3-large",
        description="Azure OpenAI embedding deployment name",
    )

    # GCP Vertex AI
    GCP_PROJECT_ID: str | None = Field(default=None, description="GCP Project ID")
    GCP_LOCATION: str = Field(default="us-central1", description="GCP Vertex AI region")
    GCP_EMBEDDING_MODEL: str = Field(
        default="text-embedding-005", description="GCP Vertex AI embedding model name"
    )

    # -------------------------------------------------------------------------
    # Cloud Object Storage
    # -------------------------------------------------------------------------
    STORAGE_PROVIDER: str = Field(
        default="local",
        description="Object storage backend: 'local', 's3', 'azure_blob', or 'gcs'",
    )
    STORAGE_BUCKET: str = Field(
        default="mentera-uploads", description="Cloud storage bucket / container name"
    )
    STORAGE_PREFIX: str = Field(default="uploads/", description="Key prefix for all stored objects")

    # -------------------------------------------------------------------------
    # Upload & Ingestion
    # -------------------------------------------------------------------------
    UPLOAD_DIR: str = Field(
        default="/tmp/mentera_uploads",  # nosec B108
        description="Local staging directory for downloaded files during ingestion",
    )
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=100, description="Maximum allowed upload file size in megabytes"
    )
    ALLOWED_FILE_EXTENSIONS: list[str] = Field(
        default=[".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff"],
        description="Permitted file extensions for document ingestion",
    )

    # -------------------------------------------------------------------------
    # Multi-Tenancy
    # -------------------------------------------------------------------------
    DEFAULT_TENANT_ID: str = Field(
        default="default", description="Fallback tenant_id when not provided"
    )

    # -------------------------------------------------------------------------
    # Chunking Defaults
    # -------------------------------------------------------------------------
    DEFAULT_CHUNK_SIZE: int = Field(default=500, description="Default chunk size in characters")
    DEFAULT_CHUNK_OVERLAP: int = Field(
        default=50, description="Default chunk overlap in characters"
    )


# Global singleton instance
settings = Settings()
