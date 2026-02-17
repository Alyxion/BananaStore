"""NiceGUI custom component wrapping the BananaStore widget.

Usage::

    from app.components import BananaStoreWidget

    BananaStoreWidget(token='...', lang='de', fallback='en')

The component is fully self-contained: it loads its own CSS, JS, and
i18n resources from the ``/static/`` mount.  No ``add_head_html`` or
``add_body_html`` calls required.
"""

from pathlib import Path

from nicegui.element import Element

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
WIDGET_HTML = (PROJ_ROOT / 'static' / 'widget.html').read_text()


class BananaStoreWidget(Element, component='bananastore.js'):
    """Embeddable BananaStore image-generation widget."""

    def __init__(
        self,
        token: str,
        lang: str = 'en',
        fallback: str = 'en',
    ) -> None:
        super().__init__()
        self._props['token'] = token
        self._props['lang'] = lang
        self._props['fallback'] = fallback
        self._props['widget_html'] = WIDGET_HTML
