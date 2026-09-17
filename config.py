from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    database_url: str

    # Ollama
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "qwen3-embedding:0.6b"

    # langfuse:
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # app
    app_name: str = "Nescio Semantic Memory API"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()