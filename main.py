"""
main.py - SentiFi FastAPI application entry point
Mounts the Dash dashboard and all API routes.
Render-ready: reads PORT from environment.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

from app.db import init_db
from app.api.routes import router
from app.dashboard.app import dash_app

load_dotenv()

app = FastAPI(
    title="SentiFi API",
    description="AI-powered stock market sentiment analysis using FinBERT + Reddit + Twitter",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all for demo; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount FastAPI routes
app.include_router(router, tags=["SentiFi"])

# Mount Dash dashboard at /dashboard
app.mount("/dashboard", WSGIMiddleware(dash_app.server))


@app.on_event("startup")
def on_startup():
    init_db()
    print("[SentiFi] Database initialized.")
    print("[SentiFi] Server running. Visit /docs for API docs or /dashboard for UI.")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
