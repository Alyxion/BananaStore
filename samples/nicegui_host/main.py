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

# Mount BananaStore static assets + WebSocket endpoint
app.mount('/static', StaticFiles(directory=str(PROJ_ROOT / 'static')), name='banana-static')

from app.ws import ws_endpoint  # noqa: E402
from app.session import registry  # noqa: E402
from app.costs import tracker as cost_tracker  # noqa: E402

app.add_api_websocket_route('/ws', ws_endpoint)

# Start session cleanup
@app.on_event('startup')
async def _start_cleanup():
    registry.start_cleanup()

@app.on_event('shutdown')
async def _stop_cleanup():
    registry.stop_cleanup()

# Load widget HTML fragment once at startup
WIDGET_HTML = (PROJ_ROOT / 'static' / 'widget.html').read_text()

# -- Budget defaults ----------------------------------------------------------

DEFAULT_BUDGET_USD = 5.00
cost_tracker.limit_usd = DEFAULT_BUDGET_USD

CATEGORY_META: dict[str, tuple[str, str]] = {
    'image_generation': ('ph ph-image', 'Images'),
    'prompt':           ('ph ph-chat-text', 'Summaries'),
    'voice_input':      ('ph ph-microphone', 'Speech to Text'),
    'voice_output':     ('ph ph-speaker-high', 'Text to Speech'),
    'image_input':      ('ph ph-eye', 'Image Analysis'),
}


# -- Dashboard page ------------------------------------------------------------

@ui.page('/')
async def dashboard():
    # Create a session and get token for this page load
    session = await registry.create_session()
    token = session.token

    # BananaStore CSS + Phosphor Icons
    ui.add_head_html(
        '<link rel="stylesheet" href="/static/styles.css">'
        '<link rel="stylesheet" href="/static/fonts/phosphor/style.css">'
    )

    # Inject the token as a meta tag
    ui.add_head_html(f'<meta name="bs-token" content="{token}">')

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
          top: 0; left: 0; right: 0; bottom: 0;
          width: 100%; height: 100%;
          border-radius: 0; margin: 0;
          z-index: 6000;
        }
      }

      /* Budget panel */
      .budget-progress-track {
        width: 100%; height: 8px;
        background: rgba(168, 115, 56, 0.15);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 12px;
      }
      .budget-progress-fill {
        height: 100%; border-radius: 4px;
        transition: width 0.6s ease, background-color 0.6s ease;
      }
      .category-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(168, 115, 56, 0.12);
        font-size: 14px; color: #816649;
      }
      .category-row:last-child { border-bottom: none; }
      .category-icon { margin-right: 8px; font-size: 16px; }
      .charge-entry {
        padding: 6px 0;
        border-bottom: 1px solid rgba(168, 115, 56, 0.10);
        font-size: 13px; color: #816649;
      }
      .charge-entry:last-child { border-bottom: none; }
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

    # -- Dashboard content (live budget tracking) --------------------------------

    with ui.element('div').classes('dashboard-content').style(
        'padding: 24px; width: 100%;'
        'font-family: "Avenir Next", "Trebuchet MS", "Gill Sans", sans-serif;'
    ):
        # --- Budget Overview Card ---
        with ui.element('div').classes('banana-card').style('margin-bottom: 16px;'):
            ui.label('Session Budget').style(
                'font-size: 17px; font-weight: 700; margin-bottom: 4px; color: #5a3e1b;'
            )
            remaining_value = ui.label(f'${DEFAULT_BUDGET_USD:.2f}')
            remaining_value.style('font-size: 32px; font-weight: 700; color: #43a047;')
            ui.label('remaining').style(
                'font-size: 13px; color: #816649; margin-top: -4px;'
            )

            with ui.element('div').classes('budget-progress-track'):
                bar_fill = ui.element('div').classes('budget-progress-fill')
                bar_fill.style('width: 0%; background-color: #43a047;')

            spent_label = ui.label(f'$0.0000 of ${DEFAULT_BUDGET_USD:.2f} used')
            spent_label.style(
                'font-size: 13px; color: #816649; margin-top: 8px;'
                'font-variant-numeric: tabular-nums;'
            )

        # --- Spending Breakdown Card ---
        with ui.element('div').classes('banana-card').style('margin-bottom: 16px;'):
            ui.label('Breakdown').style(
                'font-size: 17px; font-weight: 700; margin-bottom: 8px; color: #5a3e1b;'
            )
            cat_labels: dict[str, ui.label] = {}
            for cat_key, (icon_cls, display_name) in CATEGORY_META.items():
                with ui.element('div').classes('category-row'):
                    ui.html(
                        f'<span><i class="{icon_cls} category-icon"></i>'
                        f'{display_name}</span>'
                    )
                    lbl = ui.label('$0.0000')
                    lbl.style(
                        'font-weight: 600; font-variant-numeric: tabular-nums;'
                    )
                    cat_labels[cat_key] = lbl

        # --- Recent Charges Card ---
        with ui.element('div').classes('banana-card'):
            ui.label('Recent Charges').style(
                'font-size: 17px; font-weight: 700; margin-bottom: 8px; color: #5a3e1b;'
            )
            charges_container = ui.element('div').style(
                'max-height: 200px; overflow-y: auto;'
            )
            no_charges = ui.label('No charges yet').style(
                'font-size: 13px; color: #a08a6e; font-style: italic;'
            )

        # --- Live update timer (polls the module-level cost tracker) ---
        prev_count = {'n': 0}

        def update_budget():
            spent = cost_tracker.total_usd
            budget = cost_tracker.limit_usd or DEFAULT_BUDGET_USD
            remaining = max(0.0, budget - spent)
            pct = min(spent / budget * 100, 100) if budget > 0 else 0

            if remaining < budget * 0.1:
                color = '#e53935'
            elif remaining < budget * 0.3:
                color = '#ee7f2f'
            else:
                color = '#43a047'

            remaining_value.text = f'${remaining:.2f}'
            remaining_value.style(
                f'font-size: 32px; font-weight: 700; color: {color};'
            )
            bar_fill.style(
                f'width: {pct:.1f}%; background-color: {color};'
            )
            spent_label.text = f'${spent:.4f} of ${budget:.2f} used'

            by_cat = cost_tracker.totals_by_category()
            for cat_key, lbl in cat_labels.items():
                lbl.text = f'${by_cat.get(cat_key, 0):.4f}'

            entries = cost_tracker.entries
            if len(entries) != prev_count['n']:
                prev_count['n'] = len(entries)
                no_charges.set_visibility(len(entries) == 0)
                charges_container.clear()
                for entry in reversed(entries[-10:]):
                    meta = CATEGORY_META.get(
                        entry.category, ('ph ph-receipt', entry.category),
                    )
                    with charges_container:
                        ui.html(
                            f'<div class="charge-entry">'
                            f'<i class="{meta[0]}" style="margin-right:6px"></i>'
                            f'{meta[1]} · {entry.provider} · '
                            f'<strong>${entry.cost_usd:.4f}</strong>'
                            f'</div>'
                        )

        ui.timer(1.5, update_budget)

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
