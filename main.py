import os, sys, threading, time, webbrowser
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

def open_browser(port: int):
    time.sleep(1.4)
    webbrowser.open(f"http://127.0.0.1:{port}")

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    host = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("PORT", "8000"))
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    uvicorn.run(
        "backend.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
