"""Live browser view of the fly brain playing a game.

brain.py calls Monitor.publish() once per game frame from the main thread. A small HTTP
server on a background thread serves the page in web/ and streams every frame to it over
Server-Sent Events, so there are no dependencies beyond numpy and the standard library.

Per frame the page receives which neurons fired (a bitmap over the neurons that have a 3D
position), how many fired in each region, and the game / motor state. The neuron layout is
sent once, as binary files, when the page loads.
"""

import base64
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

WEB_DIR = Path(__file__).with_name("web")


def _sprites():
    """name -> path of the original Flappy Bird sprites shipped with flappy_bird_gymnasium."""
    import flappy_bird_gymnasium

    folder = Path(flappy_bird_gymnasium.__file__).parent / "assets" / "sprites"
    return {p.name: p for p in folder.glob("*.png")}


# Strongest non-loom inputs drawn per DNp01, for context next to the loom pathway.
OTHER_INPUTS = 40


def pathway(brain, loom_cells, dnp01, rank, n_other=OTHER_INPUTS):
    """Synapses onto each DNp01: every loom-cell input, plus its strongest other inputs.

    Returns [pre rank, index into dnp01, weight, kind] rows; kind 0 = loom cell, 1 = other.
    brain.weights is CSC (columns = presynaptic), so an input is an entry whose row is a DNp01.
    """
    entries = np.flatnonzero(np.isin(brain.indices, dnp01))
    post = brain.indices[entries]
    pre = np.searchsorted(brain.indptr, entries, side="right") - 1
    weight = brain.weights[entries]
    is_loom = np.isin(pre, loom_cells)

    edges = []
    for k, dn in enumerate(dnp01):
        onto = (post == dn) & (rank[pre] >= 0)
        loom = onto & is_loom
        other = np.flatnonzero(onto & ~is_loom)
        other = other[np.argsort(-np.abs(weight[other]))[:n_other]]
        for kind, sel in ((0, np.flatnonzero(loom)), (1, other)):
            edges += [[int(rank[pre[i]]), k, float(weight[i]), kind] for i in sel]
    return edges

# (name, colour). The page indexes into this list, so keep the order stable.
GROUPS = [
    ("Optic lobe", "#4f8cff"),
    ("Visual projection", "#22d3ee"),
    ("Central brain", "#a78bfa"),
    ("Descending", "#fbbf24"),
    ("Ascending", "#34d399"),
    ("Ventral nerve cord", "#fb7185"),
    ("Other", "#94a3b8"),
]


def group_of(superclass: str) -> int:
    if superclass.startswith("ol_") or superclass == "visual_centrifugal":
        return 0
    if superclass.startswith("visual_projection"):
        return 1
    if superclass.startswith("cb_"):
        return 2
    if "descending" in superclass:
        return 3
    if "ascending" in superclass:
        return 4
    if superclass.startswith("vnc_"):
        return 5
    return 6


class _Server(ThreadingHTTPServer):
    # The default backlog of 5 is too small: a browser opens several connections at once
    # while the game loop keeps the interpreter busy, and the overflow gets reset.
    request_queue_size = 128


class Monitor:
    def __init__(self, brain, loom_cells, dnp01, settings=None, port=8765, open_browser=True):
        self.n = brain.n
        self.dt = float(brain.dt)
        self.settings = settings or {}
        self.dnp01 = np.asarray(dnp01)

        # Only neurons with a reconstructed position can be drawn (mostly not sensory ones).
        # `rank` maps a neuron id to its index among the drawable ones, or -1.
        pos = np.asarray(brain.positions, dtype=np.float64)
        drawable = ~np.isnan(pos).any(axis=1)
        self.rank = np.full(self.n, -1, np.int64)
        self.rank[drawable] = np.arange(drawable.sum())
        self.m = int(drawable.sum())

        superclass = np.asarray(brain.superclass).astype(str)
        names, inverse = np.unique(superclass, return_inverse=True)
        self.group = np.array([group_of(s) for s in names], np.uint8)[inverse]

        # Axes: brain -> nerve cord along X, left-right along Y, depth along Z; scaled so
        # the long axis spans -1..1. The page's screen space (right, up, into the screen) is
        # left-handed, so the left-right axis is negated; otherwise the fly is drawn mirrored
        # (seen from above with the head on the left, its left side must be at the bottom).
        p = pos[drawable]
        lo, hi = p.min(axis=0), p.max(axis=0)
        mid, half = (lo + hi) / 2, (hi[2] - lo[2]) / 2
        layout = np.column_stack([(p[:, 2] - mid[2]) / half, -(p[:, 0] - mid[0]) / half, (p[:, 1] - mid[1]) / half])
        self._positions = layout.astype("<f4").tobytes()
        self._groups = self.group[drawable].tobytes()

        side = np.asarray(brain.side).astype(str)
        loom_rank = self.rank[np.asarray(loom_cells)]
        keep = loom_rank >= 0
        self._meta = json.dumps(
            {
                "n": self.n,
                "m": self.m,
                "dt": self.dt,
                "settings": self.settings,
                "groups": [
                    {"name": name, "color": color, "size": int((self.group == i).sum())}
                    for i, (name, color) in enumerate(GROUPS)
                ],
                "loom": loom_rank[keep].tolist(),
                "loomSide": side[np.asarray(loom_cells)][keep].tolist(),
                "loomTotal": int(len(loom_cells)),
                "dn": [{"idx": int(self.rank[i]), "side": str(side[i])} for i in self.dnp01],
                "paths": pathway(brain, np.asarray(loom_cells), self.dnp01, self.rank),
            }
        ).encode()

        self._sprites = _sprites()

        self._cond = threading.Condition()
        self._seq = 0
        self._payload = b""

        self.server = self._bind(port)
        self.server.monitor = self
        self.port = self.server.server_address[1]
        self.url = f"http://localhost:{self.port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print("Brain view:", self.url)
        if open_browser:
            webbrowser.open(self.url)

    @staticmethod
    def _bind(port):
        for candidate in range(port, port + 10):
            try:
                return _Server(("127.0.0.1", candidate), _Handler)
            except OSError:
                continue
        raise OSError(f"no free port in {port}..{port + 9}")

    def publish(self, fired, **fields):
        """Send one frame. `fired` = ids of the neurons that fired; fields go to the page as-is."""
        fired = np.asarray(fired, dtype=np.int64)
        drawn = self.rank[fired]
        mask = np.zeros(self.m, bool)
        mask[drawn[drawn >= 0]] = True

        fields["fired"] = int(len(fired))
        fields["groups"] = np.bincount(self.group[fired], minlength=len(GROUPS)).tolist()
        fields["dnFired"] = np.isin(self.dnp01, fired).astype(int).tolist()
        fields["bits"] = base64.b64encode(np.packbits(mask).tobytes()).decode()
        payload = json.dumps(fields, separators=(",", ":")).encode()

        with self._cond:
            self._payload = payload
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, last_seq, timeout):
        """Block until a frame newer than last_seq exists. Returns (payload, seq), payload None on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != last_seq, timeout):
                return None, last_seq
            return self._payload, self._seq

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class _Handler(BaseHTTPRequestHandler):
    # Keep-alive matters on Windows: when the server closes the socket right after a large
    # response, the tail of it (positions.bin is 1.6 MB) can be lost to a connection reset.
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        monitor = self.server.monitor
        route = self.path.split("?")[0]
        try:
            if route in ("/", "/index.html"):
                self._send((WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif route == "/meta.json":
                self._send(monitor._meta, "application/json")
            elif route == "/positions.bin":
                self._send(monitor._positions, "application/octet-stream")
            elif route == "/groups.bin":
                self._send(monitor._groups, "application/octet-stream")
            elif route == "/events":
                self._stream(monitor)
            elif route.startswith("/sprites/") and route[len("/sprites/"):] in monitor._sprites:
                self._send(monitor._sprites[route[len("/sprites/"):]].read_bytes(), "image/png")
            else:
                self.send_error(404)
        except (ConnectionError, OSError):
            pass  # the browser went away mid-response

    def _stream(self, monitor):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")  # no length: the stream runs until the page leaves
        self.end_headers()
        self.close_connection = True
        seq = 0
        while True:
            payload, seq = monitor.wait_frame(seq, timeout=10)
            self.wfile.write(b": keep-alive\n\n" if payload is None else b"data: " + payload + b"\n\n")
            self.wfile.flush()
