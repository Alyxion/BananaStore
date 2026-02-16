"""
NiceGUI sample app demonstrating BananaStore embedded as a direct widget.

BananaStore's HTML fragment is injected into the page — no iframe needed.
API keys are forwarded via the shared ``app.config.settings`` singleton.

Prerequisites:
    Start HTTPS proxy:        docker compose up
    Then run this app:        poetry run python samples/nicegui_host/main.py
    Open https://localhost:8453
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from nicegui import app, ui
from starlette.staticfiles import StaticFiles

# -- Bootstrap BananaStore inside NiceGUI -------------------------------------

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

# Load .env from project root so API keys are available
load_dotenv(PROJ_ROOT / '.env')

# Populate BananaStore config singleton from environment
from app.config import settings  # noqa: E402

for key in ('OPENAI_API_KEY', 'GOOGLE_API_KEY', 'ANTHROPIC_API_KEY', 'COST_LIMIT_USD'):
    value = os.getenv(key)
    if value is not None:
        if key == 'COST_LIMIT_USD':
            settings.COST_LIMIT_USD = float(value)
        else:
            setattr(settings, key, value)

# Mount BananaStore static assets + API routes
app.mount('/static', StaticFiles(directory=str(PROJ_ROOT / 'static')), name='banana-static')

from app.main import app as banana_app  # noqa: E402

for route in banana_app.routes:
    if hasattr(route, 'path') and route.path.startswith('/api/'):
        app.routes.insert(0, route)

# Load widget HTML fragment once at startup
WIDGET_HTML = (PROJ_ROOT / 'static' / 'widget.html').read_text()


# -- Dashboard page ------------------------------------------------------------

@ui.page('/')
def dashboard():
    # BananaStore CSS + Phosphor Icons
    ui.add_head_html(
        '<link rel="stylesheet" href="/static/styles.css">'
        '<link rel="stylesheet" href="/static/fonts/phosphor/style.css">'
    )

    ui.add_head_html('''
    <style>
      .banana-card {
        background: rgba(255, 249, 236, 0.74);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(168, 115, 56, 0.32);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 28px rgba(81, 34, 7, 0.14);
      }
      .stat-value { font-size: 32px; font-weight: 700; color: #ee7f2f; }
      .stat-label { font-size: 13px; color: #816649; margin-top: 4px; }

      /* Keep dashboard content from flowing behind the store panel */
      .dashboard-content {
        max-width: calc((100vw - min(1220px, calc(100vw - 48px))) / 2 - 16px);
        min-width: 280px;
      }
      @media (max-width: 900px) {
        .dashboard-content { display: none; }
      }

      /* BananaStore floating panel — centered */
      .store-panel {
        position: fixed;
        top: 60px;
        left: 0;
        right: 0;
        width: min(1220px, calc(100vw - 48px));
        height: calc(100vh - 84px);
        margin: 0 auto;
        border-radius: 16px;
        z-index: 6000;
      }

      /* Mobile: fullscreen, dock to 0,0, above NiceGUI header */
      @media (max-width: 900px) {
        .store-panel {
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          width: 100%;
          height: 100%;
          border-radius: 0;
          margin: 0;
          z-index: 6000;
        }
      }
    </style>
    ''')

    ui.query('body').style(
        'background: radial-gradient(120% 120% at 12% 12%,'
        ' #ffe79f 0%, #ffbf63 32%, #f48a3a 62%, #cf5d24 100%);'
        'background-attachment: fixed;'
    )
    ui.query('.nicegui-content').style('padding: 0;')

    # -- Header ----------------------------------------------------------------

    with ui.header().style(
        'background: linear-gradient(135deg, #f48a3a 0%, #ee7f2f 50%, #cc611d 100%);'
        'box-shadow: 0 4px 16px rgba(81, 34, 7, 0.18);'
    ):
        ui.label('Banana Dashboard').style('font-size: 20px; font-weight: 700;')

    # -- Dashboard content --------------------------------------------------------

    with ui.element('div').classes('dashboard-content').style(
        'padding: 24px; width: 100%;'
        'font-family: "Avenir Next", "Trebuchet MS", "Gill Sans", sans-serif;'
    ):
        # Stats row
        with ui.row().style('gap: 16px; margin-bottom: 20px; flex-wrap: wrap; width: 100%;'):
            for value, label in [
                ('1,247', 'Images Generated'),
                ('38', 'Active Sessions'),
                ('99.7%', 'Uptime'),
                ('4.2s', 'Avg Response'),
            ]:
                with ui.element('div').classes('banana-card').style(
                    'flex: 1; min-width: 140px; text-align: center; padding: 16px;'
                ):
                    ui.html(f'<div class="stat-value">{value}</div>'
                            f'<div class="stat-label">{label}</div>')

        # Activity card
        with ui.element('div').classes('banana-card'):
            ui.label('Recent Activity').style(
                'font-size: 17px; font-weight: 700; margin-bottom: 12px;'
            )
            for entry in [
                'Landscape generated — 2 min ago',
                'Portrait style applied — 8 min ago',
                'Batch export completed — 15 min ago',
                'New template saved — 23 min ago',
                'SVG vector created — 31 min ago',
            ]:
                ui.label(entry).style(
                    'padding: 8px 0; border-bottom: 1px solid rgba(168,115,56,0.15);'
                    'color: #816649; font-size: 14px;'
                )

    # Embedded BananaStore widget — direct HTML, no iframe
    ui.html(
        f'<div class="store-panel">'
        f'<div class="banana-store">{WIDGET_HTML}</div>'
        f'</div>'
    )

    # Load BananaStore JS (auto-inits on .banana-store container)
    ui.add_body_html('<script src="/static/app.js"></script>')


ui.run(port=8070, title='Banana Dashboard', show=False, reload=True,
       uvicorn_reload_includes='*.js,*.css')
