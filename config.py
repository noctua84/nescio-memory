from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    database_url: str

    # Ollama
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "qwen3-embedding:0.6b"
    embedding_dimension: int = 384

    # langfuse:
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # app
    app_name: str = "Nescio Semantic Memory API"
    log_level: str = "INFO"

    # chunks
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # server
    host: str = "localhost"
    port: int = 8080

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()