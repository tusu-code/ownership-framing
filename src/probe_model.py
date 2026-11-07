"""Exit 0 if the named model in config.yaml answers a one-line request, else 1."""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import Client, load_config  # noqa: E402
import time

async def main(name):
    cfg = load_config()
    c = Client(name, cfg["models"][name], cfg.get("defaults", {}))
    try:
        out = await c.chat([{"role": "user", "content": "Reply with the single word OK."}], 50)
        ok = bool((out["text"] or "").strip())
    except Exception as e:  # noqa: BLE001
        print(time.strftime("%F %T"), name, "DOWN", str(e)[:100]); ok = False
    await c.http.aclose()
    if ok: print(time.strftime("%F %T"), name, "UP")
    return ok

sys.exit(0 if asyncio.run(main(sys.argv[1])) else 1)
