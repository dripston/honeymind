from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.victim_api.api.endpoints import router as api_router
from backend.victim_api.ml.inference import predictor
from backend.victim_api.core.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    logger.info("Starting up Victim API...")
    success = predictor.load_model()
    if not success:
        logger.warning("API started but ML model is NOT loaded. The /predict endpoint will fail.")
    yield
    # Shutdown: Clean up resources if necessary
    logger.info("Shutting down Victim API...")

app = FastAPI(
    title="Victim Fraud Detection API",
    description="A realistic machine learning API for detecting credit card fraud. This API will later be intercepted by HoneyMind.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.victim_api.main:app", host="127.0.0.1", port=8000, reload=True)
