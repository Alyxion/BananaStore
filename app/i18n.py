"""Project-level i18n — reads from ``static/locales/<lang>.json``.

The same locale files are served to the JS frontend via the static mount
and read on the Python side for server-rendered UI (e.g. NiceGUI).

Fallback chain:  overrides → lang → fallback → English defaults.

Usage::

    from app.i18n import I18n

    t = I18n('de')                        # German, English fallback
    t = I18n('de', fallback='fr')         # German, French fallback, then English
    t = I18n(overrides={'remaining': 'verbleibend'})  # English + one override

    t('session_budget')                   # "Sitzungsbudget"
    t('spent_of_budget',                  # "$0.42 von $5.00 verbraucht"
      spent='$0.42', budget='$5.00')
"""

import json
from pathlib import Path

LOCALES_DIR = Path(__file__).resolve().parent.parent / 'static' / 'locales'

# Baked-in English defaults — the system works even without any JSON files.
DEFAULTS: dict[str, str] = {
    'bs.dashboard_title':    'Banana Dashboard',
    'bs.session_budget':     'Session Budget',
    'bs.remaining':          'remaining',
    'bs.spent_of_budget':    '{spent} of {budget} used',
    'bs.breakdown':          'Breakdown',
    'bs.cat_images':         'Images',
    'bs.cat_summaries':      'Summaries',
    'bs.cat_stt':            'Speech to Text',
    'bs.cat_tts':            'Text to Speech',
    'bs.cat_image_analysis': 'Image Analysis',
    'bs.recent_charges':     'Recent Charges',
    'bs.no_charges':         'No charges yet',
    'bs.charge_entry':       '{category} \u00b7 {provider} \u00b7 {cost}',
}


def _load_json(path: Path) -> dict[str, str]:
    if path.is_file():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


class I18n:
    """Simple key→string translator with ``.format()`` interpolation."""

    def __init__(
        self,
        lang: str = 'en',
        fallback: str = 'en',
        locales_dir: Path | str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> None:
        search = Path(locales_dir) if locales_dir else LOCALES_DIR

        # Start with baked-in English
        self._strings: dict[str, str] = dict(DEFAULTS)

        # Layer on en.json (may contain keys not yet in DEFAULTS)
        self._strings.update(_load_json(search / 'en.json'))

        # Layer on fallback language (if not English)
        if fallback != 'en':
            self._strings.update(_load_json(search / f'{fallback}.json'))

        # Layer on target language
        if lang not in ('en', fallback):
            self._strings.update(_load_json(search / f'{lang}.json'))

        # Host-specific overrides win
        if overrides:
            self._strings.update(overrides)

    def __call__(self, key: str, **kwargs: str) -> str:
        template = self._strings.get(key, key)
        return template.format(**kwargs) if kwargs else template
