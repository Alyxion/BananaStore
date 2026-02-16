from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.costs import SpendingLimitExceeded, tracker
from app.session import registry
from app.ws import ws_endpoint


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv()
    limit = settings.get("COST_LIMIT_USD")
    if limit:
        tracker.limit_usd = float(limit)
    registry.start_cleanup()
    yield
    registry.stop_cleanup()


app = FastAPI(title="BananaStore", lifespan=lifespan)


@app.exception_handler(SpendingLimitExceeded)
async def spending_limit_handler(_request: Request, exc: SpendingLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc), "limit": exc.limit, "current": exc.current, "attempted": exc.attempted},
    )

app.websocket("/ws")(ws_endpoint)

_APP_DIR = Path(__file__).resolve().parent.parent

app.mount("/static", StaticFiles(directory=_APP_DIR / "static"), name="static")


def enable_standalone() -> None:
    """Register GET / to serve index.html with auto-created anonymous sessions.

    Disabled by default so hosts (e.g. NiceGUI) that manage their own
    sessions and tokens are not exposed to an unauthenticated endpoint.
    Call this explicitly for standalone deployments.
    """
    index_html = (_APP_DIR / "static" / "index.html").read_text()

    @app.get("/")
    async def root() -> HTMLResponse:
        session = await registry.create_session()
        html = index_html.replace(
            "</head>",
            f'<meta name="bs-token" content="{session.token}">\n  </head>',
            1,
        )
        return HTMLResponse(html)
