export default {
  template: '<div ref="root" class="banana-store"></div>',
  props: {
    token: String,
    lang: { type: String, default: 'en' },
    fallback: { type: String, default: 'en' },
    widget_html: String,
  },
  mounted() {
    this._loadCSS('/static/styles.css');
    this._loadCSS('/static/fonts/phosphor/style.css');

    this._setMeta('bs-token', this.token);
    this._setMeta('bs-lang', this.lang);
    this._setMeta('bs-lang-fallback', this.fallback);

    this.$refs.root.innerHTML = this.widget_html;

    this._ensureScript('/static/i18n.js')
      .then(() => this._ensureScript('/static/app.js'))
      .then(() => {
        const init = () => {
          if (window.BananaStore) window.BananaStore.init(this.$refs.root);
        };
        if (window.bsI18n) {
          window.bsI18n.ready.then(init);
        } else {
          init();
        }
      });
  },
  methods: {
    _loadCSS(href) {
      if (document.querySelector(`link[href="${href}"]`)) return;
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = href;
      document.head.appendChild(link);
    },
    _ensureScript(src) {
      return new Promise((resolve) => {
        if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
        const el = document.createElement('script');
        el.src = src;
        el.onload = resolve;
        el.onerror = resolve;
        document.body.appendChild(el);
      });
    },
    _setMeta(name, content) {
      let meta = document.querySelector(`meta[name="${name}"]`);
      if (!meta) {
        meta = document.createElement('meta');
        meta.name = name;
        document.head.appendChild(meta);
      }
      meta.content = content;
    },
  },
};
