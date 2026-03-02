from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Image-to-Graph API"
    API_V1_STR: str = "/api/v1"
    UPLOAD_DIR: str = "static/uploads"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./app.db"
    
    # JWT Auth
    SECRET_KEY: str = "super_secret_key_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    
    class Config:
        env_file = ".env"

settings = Settings()
