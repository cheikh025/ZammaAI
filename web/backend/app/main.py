from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import games, health
from .services.session import session_manager

# Path to the compiled frontend
DIST = Path(__file__).parents[2] / "frontend" / "dist"


# ---------------------------------------------------------------------------
# Lifespan: start TTL cleanup task on boot, cancel on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(session_manager.cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Khreibga Zero API",
    description="Backend game API for the Khreibga web interface.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server and any same-origin prod URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes (must be registered before the static catch-all)
app.include_router(health.router, prefix="/api")
app.include_router(games.router,  prefix="/api")

# ---------------------------------------------------------------------------
# Frontend static files
# Serve the built React app from web/frontend/dist/.
# Run  cd web/frontend && npm run build  once before using this.
# ---------------------------------------------------------------------------

# JS / CSS / image assets (e.g. /assets/index-Biv8COAq.js)
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="frontend-assets")


def _index() -> FileResponse | HTMLResponse:
    if (DIST / "index.html").exists():
        return FileResponse(DIST / "index.html")
    return HTMLResponse(
        "<p>Frontend not built. Run: <code>cd web/frontend &amp;&amp; npm run build</code></p>",
        status_code=503,
    )


@app.get("/", include_in_schema=False)
async def serve_root():
    return _index()


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    return _index()
