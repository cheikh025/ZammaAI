# Web Interface — Command Reference

All commands are run from the **repo root** (`ZammaAI/`) unless stated otherwise.

---

## 1. Start the Backend

```powershell
python -m uvicorn web.backend.app.main:app --port 8000 --reload
```

| Flag | Meaning |
|---|---|
| `--port 8000` | Listen on port 8000 |
| `--reload` | Auto-restart when source files change (dev only) |

API base URL: `http://localhost:8000/api`
Interactive docs: `http://localhost:8000/docs`

---

## 2. Start the Frontend

```powershell
cd web/frontend
npm run dev
```

Opens at: `http://localhost:5173`

To return to repo root afterwards:

```powershell
cd ../..
```

---

## 3. Build the Frontend for Production (single-server mode)

```powershell
cd web/frontend
npm run build
cd ../..
python -m uvicorn web.backend.app.main:app --port 8000
```

Then open `http://localhost:8000` — the backend serves the game directly.
No second terminal needed.

Output of the build goes to `web/frontend/dist/`.

---

## 4. Port Troubleshooting (Windows)

### Find what is using a port

```powershell
netstat -ano | findstr ":8000"
```

The last column is the **PID** (process ID).

```
TCP    127.0.0.1:8000    0.0.0.0:0    LISTENING    12345
                                                    ^^^^^
                                                    PID
```

### Kill a process by PID

```powershell
taskkill /PID 12345 /F
```

`/F` forces termination. Replace `12345` with the actual PID.

### Kill all Python processes (nuclear option — closes everything)

```powershell
taskkill /IM python.exe /F
```

### Check a port is free before starting

```powershell
netstat -ano | findstr ":8000"
# No output = port is free
```

---

## 5. Run on a Different Port

If port 8000 is blocked, pick any free port (e.g. 8001):

**Backend:**
```powershell
python -m uvicorn web.backend.app.main:app --port 8001 --reload
```

**Frontend** — update `vite.config.ts` proxy target to match:
```ts
proxy: {
  '/api': 'http://localhost:8001',
}
```

Then restart `npm run dev`.

---

## 6. Install / Reinstall Dependencies

**Backend:**
```powershell
pip install fastapi "uvicorn[standard]"
```

**Frontend:**
```powershell
cd web/frontend
npm install
```

---

## 7. Stop Servers

Press `Ctrl + C` in the terminal where the server is running.

If `Ctrl + C` doesn't work (server is stuck):

```powershell
# Find the PID on port 8000
netstat -ano | findstr ":8000"

# Kill it
taskkill /PID <PID> /F
```

---

## 8. Quick Sanity Checks

### Is the backend alive?

```powershell
curl http://localhost:8000/api/health
# Expected: {"status":"ok"}
```

### Is the frontend proxying correctly?

Open `http://localhost:5173` in a browser.
Click **Start Game** — if no red error appears, the proxy is working.

### List active sessions (Python one-liner)

```powershell
python -c "from web.backend.app.services.session import session_manager; print(list(session_manager._sessions.keys()))"
```

---

## 9. Typical Dev Workflow

```
Terminal 1                          Terminal 2
──────────────────────────────────  ────────────────────────────────
python -m uvicorn \                 cd web/frontend
  web.backend.app.main:app \        npm run dev
  --port 8000 --reload
```

Then open `http://localhost:5173` in a browser.

Both servers hot-reload on file save — no restart needed for most changes.
