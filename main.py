import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
import os
from app.core.config import settings
from app.api.router import router as api_router
from app.core.database import engine, Base
from app.models.upload import UploadHistory

# Auto-create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

# CORS — cho phép bên ngoài gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/history", response_class=HTMLResponse)
async def read_history():
    with open("templates/history.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/prompt-tool", response_class=HTMLResponse)
async def read_prompt_tool():
    with open("templates/prompt_tool.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)
