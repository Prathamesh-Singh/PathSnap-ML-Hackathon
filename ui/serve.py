"""Serve the evaluator UI (runs on Laptop 2).

    python ui/serve.py            # then open http://localhost:5500
    python ui/serve.py --port 8080

Enter Laptop 1's API address (e.g. http://192.168.1.20:8000) in the page.
"""
import argparse
import functools
import http.server
import os

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5500)
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=here)
with http.server.ThreadingHTTPServer(("0.0.0.0", a.port), handler) as srv:
    print(f"PathSnap UI on http://localhost:{a.port}  (Ctrl+C to stop)")
    srv.serve_forever()
