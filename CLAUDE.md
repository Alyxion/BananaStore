# Project Rules

## External References

- **No external CDN or remote resource references allowed** — all assets (fonts, icons, CSS, JS libraries) must be self-hosted in the `static/` directory. This is a strict requirement due to tracking concerns.
- Cleanly attribute licenses of any added third-party sources (e.g., in a `LICENSES` file or inline comments).

## Development & Testing

- All apps bind to **port 8070**. Access exclusively via the HTTPS proxy.
- **BananaStore**: `poetry run uvicorn app.main:app --host 0.0.0.0 --port 8070 --reload`
- **NiceGUI sample**: `poetry run python samples/nicegui_host/main.py` (also port 8070). **Hot-reloads on file changes** (`.py`, `.js`, `.css`) — do not restart the process after editing.
- **HTTPS proxy** (self-signed): `docker compose up` → `https://localhost:8453` (nginx → port 8070)
- **Always test via HTTPS** (`https://localhost:8453`) — never open plain HTTP ports in the browser.
- **Do NOT use the `nice-vibes kill_port_8080` MCP tool** — we do not use port 8080.
- **Killing port listeners**: Only kill TCP LISTEN sockets, never all connections. Use `lsof -ti :PORT -sTCP:LISTEN | xargs kill` — a bare `lsof -ti :PORT` also matches connected clients (like the nginx proxy) and will take down unrelated services.
