from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict
from .routes import router
from ..utils.logger import logger

app = FastAPI(
    title="Credit Fraud Detection API",
    description="Real-time fraud detection system for credit card transactions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Credit Fraud Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fraud-detection"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )