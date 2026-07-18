from fastapi import FastAPI
from backend.gateway.api.endpoints import router as api_router
from backend.gateway.api.control_endpoints import router as control_router
from backend.gateway.core.logger import logger

app = FastAPI(
    title="HoneyMind Gateway",
    description="Middleware Gateway that intercepts traffic, performs threat detection, and proxies to the Target API.",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(control_router, prefix="/api/v1/control")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting HoneyMind Gateway on port 8001...")
    uvicorn.run("backend.gateway.main:app", host="127.0.0.1", port=8001, reload=True)
