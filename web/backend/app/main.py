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
    # Warm up the model at startup so the first AI move has no cold-start delay.
    # This loads torch, instantiates KhreibagaNet, reads the checkpoint, and runs
    # one dummy forward pass to trigger PyTorch kernel JIT compilation.
    def _warmup():
        try:
            import torch
            from .services.ai import _load_model
            model, device = _load_model()
            dummy = torch.zeros(1, 7, 5, 5, device=device)
            with torch.no_grad():
                model(dummy)
            print("[startup] Model warmed up.")
        except Exception as exc:
            print(f"[startup] Model warmup failed (non-fatal): {exc}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _warmup)

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
