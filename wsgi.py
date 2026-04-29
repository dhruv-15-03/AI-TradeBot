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

    import urllib.request, urllib.error

    def _ping():
        # Wait 2 min after startup so the app is fully warmed up first
        time.sleep(120)
        while True:
            try:
                url = f"{base_url}/api/status"
                urllib.request.urlopen(url, timeout=25)
                log.info("Keep-alive ping OK → %s", url)
            except urllib.error.URLError as e:
                # Timeout = server is already busy/awake — not a real failure
                reason = str(e.reason) if hasattr(e, "reason") else str(e)
                if "timed out" in reason.lower():
                    log.debug("Keep-alive ping timed out (server busy but alive)")
                else:
                    log.warning("Keep-alive ping failed: %s", reason)
            except Exception as e:
                log.warning("Keep-alive ping error: %s", e)
            time.sleep(10 * 60)   # ping every 10 minutes

    t = threading.Thread(target=_ping, daemon=True, name="KeepAlive")
    t.start()
    log.info("Keep-alive started — pinging %s every 10 min.", base_url)


start_background_refresh()
start_keep_alive()

if __name__ == "__main__":
    app.run()
