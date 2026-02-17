/**
 * Lightweight i18n for BananaStore.
 *
 * Locale JSON files live in /static/locales/<lang>.json.
 * Falls back through: overrides → lang → fallback → English.
 *
 * Auto-initialises from <meta name="bs-lang"> if present,
 * otherwise defaults to English.
 *
 * Usage:
 *   const i18n = new BsI18n('de');
 *   await i18n.ready;
 *   i18n.t('session_budget');
 *   i18n.t('spent_of_budget', { spent: '$0.42', budget: '$5.00' });
 */

class BsI18n {
  /**
   * @param {string}  lang         Target language code (default: 'en').
   * @param {string}  fallback     Fallback language code (default: 'en').
   * @param {Object}  [overrides]  Key→string overrides applied last.
   */
  constructor(lang = 'en', fallback = 'en', overrides = null) {
    this._en = {};
    this._fallback = {};
    this._lang = {};
    this._overrides = overrides || {};
    this._langCode = lang;
    this._fallbackCode = fallback;
    this.ready = this._load();
  }

  async _load() {
    // Always load English as the base
    this._en = await this._fetch('en');

    // Load fallback if it differs from English
    if (this._fallbackCode !== 'en') {
      this._fallback = await this._fetch(this._fallbackCode);
    }

    // Load target language if it differs from both
    if (this._langCode !== 'en' && this._langCode !== this._fallbackCode) {
      this._lang = await this._fetch(this._langCode);
    } else if (this._langCode !== 'en') {
      this._lang = this._fallback;
    }
  }

  async _fetch(code) {
    try {
      const resp = await fetch(`/static/locales/${encodeURIComponent(code)}.json`);
      if (resp.ok) return await resp.json();
    } catch { /* missing locale file — not an error */ }
    return {};
  }

  /**
   * Translate a key, interpolating {placeholder} tokens.
   * Lookup order: overrides → lang → fallback → en → raw key.
   */
  t(key, params) {
    let tpl = this._overrides[key]
           ?? this._lang[key]
           ?? this._fallback[key]
           ?? this._en[key]
           ?? key;
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        tpl = tpl.replaceAll(`{${k}}`, v);
      }
    }
    return tpl;
  }

  /** Merge additional overrides at runtime. */
  setOverrides(obj) {
    Object.assign(this._overrides, obj);
  }
}

/**
 * Auto-initialise a global instance from meta tags:
 *   <meta name="bs-lang"          content="de">
 *   <meta name="bs-lang-fallback" content="en">
 */
(function () {
  const langMeta     = document.querySelector('meta[name="bs-lang"]');
  const fallbackMeta = document.querySelector('meta[name="bs-lang-fallback"]');
  const lang     = (langMeta     && langMeta.content)     || 'en';
  const fallback = (fallbackMeta && fallbackMeta.content) || 'en';

  // Pick up inline overrides: <script type="application/json" id="bs-i18n-overrides">
  let overrides = null;
  const ovEl = document.getElementById('bs-i18n-overrides');
  if (ovEl) {
    try { overrides = JSON.parse(ovEl.textContent); } catch { /* ignore */ }
  }

  window.bsI18n = new BsI18n(lang, fallback, overrides);
})();
