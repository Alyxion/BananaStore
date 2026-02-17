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
from app.i18n import I18n  # noqa: E402
from app.components import BananaStoreWidget  # noqa: E402

app.add_api_websocket_route('/ws', ws_endpoint)

# Start session cleanup
@app.on_event('startup')
async def _start_cleanup():
    registry.start_cleanup()

@app.on_event('shutdown')
async def _stop_cleanup():
    registry.stop_cleanup()

# -- Budget defaults ----------------------------------------------------------

DEFAULT_BUDGET_USD = 5.00
cost_tracker.limit_usd = DEFAULT_BUDGET_USD

# i18n — set BS_LANG / BS_LANG_FALLBACK env vars, or pass overrides=
BS_LANG = os.getenv('BS_LANG', 'en')
BS_FALLBACK = os.getenv('BS_LANG_FALLBACK', 'en')
t = I18n(lang=BS_LANG, fallback=BS_FALLBACK)

CATEGORY_ICONS: dict[str, str] = {
    'image_generation': 'ph ph-image',
    'prompt':           'ph ph-chat-text',
    'voice_input':      'ph ph-microphone',
    'voice_output':     'ph ph-speaker-high',
    'image_input':      'ph ph-eye',
}

CATEGORY_I18N: dict[str, str] = {
    'image_generation': 'bs.cat_images',
    'prompt':           'bs.cat_summaries',
    'voice_input':      'bs.cat_stt',
    'voice_output':     'bs.cat_tts',
    'image_input':      'bs.cat_image_analysis',
}

# -- Reusable inline styles ---------------------------------------------------

CARD_STYLE = (
    'background: rgba(255, 249, 236, 0.74);'
    'backdrop-filter: blur(10px);'
    'border: 1px solid rgba(168, 115, 56, 0.32);'
    'border-radius: 16px;'
    'padding: 24px;'
    'box-shadow: 0 8px 28px rgba(81, 34, 7, 0.14);'
)

STORE_PANEL_STYLE = (
    'position: fixed; top: 60px; left: 0; right: 0;'
    'width: min(1220px, calc(100vw - 48px)); height: calc(100vh - 84px);'
    'margin: 0 auto; border-radius: 16px; z-index: 6000;'
)

CATEGORY_ROW_STYLE = (
    'display: flex; justify-content: space-between; align-items: center;'
    'padding: 8px 0; border-bottom: 1px solid rgba(168, 115, 56, 0.12);'
    'font-size: 14px; color: #816649;'
)

CHARGE_ENTRY_STYLE = (
    'padding: 6px 0; border-bottom: 1px solid rgba(168, 115, 56, 0.10);'
    'font-size: 13px; color: #816649;'
)

PROGRESS_TRACK_STYLE = (
    'width: 100%; height: 8px;'
    'background: rgba(168, 115, 56, 0.15);'
    'border-radius: 4px; overflow: hidden; margin-top: 12px;'
)

PROGRESS_FILL_STYLE = (
    'height: 100%; border-radius: 4px;'
    'transition: width 0.6s ease, background-color 0.6s ease;'
)


# -- Dashboard page ------------------------------------------------------------

@ui.page('/')
async def dashboard():
    session = await registry.create_session()
    token = session.token

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
        ui.label(t('bs.dashboard_title')).style('font-size: 20px; font-weight: 700;')

    # -- Dashboard content (live budget tracking) --------------------------------

    with ui.element('div').style(
        'padding: 24px; width: 100%;'
        'max-width: calc((100vw - min(1220px, calc(100vw - 48px))) / 2 - 16px);'
        'min-width: 280px;'
        'font-family: "Avenir Next", "Trebuchet MS", "Gill Sans", sans-serif;'
    ):
        # --- Budget Overview Card ---
        with ui.element('div').style(f'{CARD_STYLE} margin-bottom: 16px;'):
            ui.label(t('bs.session_budget')).style(
                'font-size: 17px; font-weight: 700; margin-bottom: 4px; color: #5a3e1b;'
            )
            remaining_value = ui.label(f'${DEFAULT_BUDGET_USD:.2f}')
            remaining_value.style('font-size: 32px; font-weight: 700; color: #43a047;')
            ui.label(t('bs.remaining')).style(
                'font-size: 13px; color: #816649; margin-top: -4px;'
            )

            with ui.element('div').style(PROGRESS_TRACK_STYLE):
                bar_fill = ui.element('div').style(
                    f'{PROGRESS_FILL_STYLE} width: 0%; background-color: #43a047;'
                )

            spent_label = ui.label(
                t('bs.spent_of_budget', spent='$0.0000', budget=f'${DEFAULT_BUDGET_USD:.2f}')
            )
            spent_label.style(
                'font-size: 13px; color: #816649; margin-top: 8px;'
                'font-variant-numeric: tabular-nums;'
            )

        # --- Spending Breakdown Card ---
        with ui.element('div').style(f'{CARD_STYLE} margin-bottom: 16px;'):
            ui.label(t('bs.breakdown')).style(
                'font-size: 17px; font-weight: 700; margin-bottom: 8px; color: #5a3e1b;'
            )
            cat_labels: dict[str, ui.label] = {}
            for cat_key, i18n_key in CATEGORY_I18N.items():
                icon_cls = CATEGORY_ICONS[cat_key]
                display_name = t(i18n_key)
                with ui.element('div').style(CATEGORY_ROW_STYLE):
                    ui.html(
                        f'<span><i class="{icon_cls}" style="margin-right: 8px; font-size: 16px;"></i>'
                        f'{display_name}</span>'
                    )
                    lbl = ui.label('$0.0000')
                    lbl.style(
                        'font-weight: 600; font-variant-numeric: tabular-nums;'
                    )
                    cat_labels[cat_key] = lbl

        # --- Recent Charges Card ---
        with ui.element('div').style(CARD_STYLE):
            ui.label(t('bs.recent_charges')).style(
                'font-size: 17px; font-weight: 700; margin-bottom: 8px; color: #5a3e1b;'
            )
            charges_container = ui.element('div').style(
                'max-height: 200px; overflow-y: auto;'
            )
            no_charges = ui.label(t('bs.no_charges')).style(
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
                f'{PROGRESS_FILL_STYLE} width: {pct:.1f}%; background-color: {color};'
            )
            spent_label.text = t(
                'bs.spent_of_budget', spent=f'${spent:.4f}', budget=f'${budget:.2f}',
            )

            by_cat = cost_tracker.totals_by_category()
            for cat_key, lbl in cat_labels.items():
                lbl.text = f'${by_cat.get(cat_key, 0):.4f}'

            entries = cost_tracker.entries
            if len(entries) != prev_count['n']:
                prev_count['n'] = len(entries)
                no_charges.set_visibility(len(entries) == 0)
                charges_container.clear()
                for entry in reversed(entries[-10:]):
                    icon = CATEGORY_ICONS.get(entry.category, 'ph ph-receipt')
                    i18n_key = CATEGORY_I18N.get(entry.category)
                    cat_name = t(i18n_key) if i18n_key else entry.category
                    label = t(
                        'bs.charge_entry',
                        category=cat_name,
                        provider=entry.provider,
                        cost=f'${entry.cost_usd:.4f}',
                    )
                    with charges_container:
                        ui.html(
                            f'<div style="{CHARGE_ENTRY_STYLE}">'
                            f'<i class="{icon}" style="margin-right: 6px;"></i>'
                            f'{label}'
                            f'</div>'
                        )

        ui.timer(1.5, update_budget)

    # -- Embedded BananaStore widget (self-contained custom component) ----------

    with ui.element('div').style(STORE_PANEL_STYLE):
        BananaStoreWidget(token=token, lang=BS_LANG, fallback=BS_FALLBACK)


ui.run(port=8070, title=t('bs.dashboard_title'), show=False, reload=True,
       uvicorn_reload_includes='*.js,*.css')
