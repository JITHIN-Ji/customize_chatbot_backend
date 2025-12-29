from dotenv import load_dotenv
load_dotenv()
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles 
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware   
from app.db.session import engine
from app.db.base_class import Base
from app.models import chatbot, user # Assuming you also have a user.py model
from app.api.endpoints import router as api_router
from app.routes import auth
from app.core.config import settings
import logging

Base.metadata.create_all(bind=engine)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="A RAG‑powered API for interacting with documents."
)


app.add_middleware(
    CORSMiddleware,
     allow_origins=[
        settings.CLIENT_ORIGIN_URL or "http://localhost:3000",
        "null" 
    ], 

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("APP_SECRET"),
    same_site="lax",
    https_only=False,
    max_age=3600,
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")



app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(auth.router, prefix=settings.API_V1_STR) 

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    os.makedirs("uploads/avatars", exist_ok=True)
    

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}. Visit {settings.API_V1_STR}/docs for documentation."}
