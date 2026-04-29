"""WSGI entry point for Render.com / gunicorn."""
import sys, os, threading, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, start_background_refresh

log = logging.getLogger(__name__)


def start_keep_alive():
    """
    Ping our own /api/status every 10 minutes so Render free tier never sleeps.
    Only activates when RENDER_EXTERNAL_URL is set (i.e. running on Render).
    Skipped silently in local dev.
    """
    base_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not base_url:
        return   # local dev — no ping needed

    import urllib.request

    def _ping():
        # Wait 2 min after startup so the app is fully warmed up first
        time.sleep(120)
        while True:
            try:
                url = f"{base_url}/api/status"
                urllib.request.urlopen(url, timeout=10)
                log.info("Keep-alive ping OK → %s", url)
            except Exception as e:
                log.warning("Keep-alive ping failed: %s", e)
            time.sleep(10 * 60)   # ping every 10 minutes

    t = threading.Thread(target=_ping, daemon=True, name="KeepAlive")
    t.start()
    log.info("Keep-alive started — pinging %s every 10 min.", base_url)


start_background_refresh()
start_keep_alive()

if __name__ == "__main__":
    app.run()
