from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.costs import SpendingLimitExceeded, tracker
from app.routes import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv()
    limit = settings.get("COST_LIMIT_USD")
    if limit:
        tracker.limit_usd = float(limit)
    yield


app = FastAPI(title="BananaStore", lifespan=lifespan)


@app.exception_handler(SpendingLimitExceeded)
async def spending_limit_handler(_request: Request, exc: SpendingLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc), "limit": exc.limit, "current": exc.current, "attempted": exc.attempted},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

_APP_DIR = Path(__file__).resolve().parent.parent


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(_APP_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=_APP_DIR / "static"), name="static")
