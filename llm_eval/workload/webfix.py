"""Canned web pages for the harness's webfetch: recorded once, replayed identically every run.

An HTTP proxy on host loopback, reached from the sandbox through a relay. Plain ``http://`` URLs
are looked up directly; ``CONNECT`` to a canned host is answered with TLS under a run-local CA
(trusted inside the sandbox only, via ``NODE_EXTRA_CA_CERTS``), then the request inside is looked
up. Anything not canned gets an error page and is counted as a miss, so a run is repeatable and
the benchmark never touches the internet.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import ssl
import subprocess
import threading
from urllib.parse import urlsplit

PROXY_INSIDE = "127.0.0.1:3128"
# Hosts the harness itself reaches for at startup; refused like any other, but not task misses.
HARNESS_HOSTS = ("registry.npmjs.org", "models.dev", "opencode.ai")


def make_ca(directory, hosts):
    """A throwaway CA and one leaf certificate for exactly ``hosts``; returns (ca, cert, key)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    def openssl(*args):
        out = subprocess.run(["openssl", *args], capture_output=True, text=True, timeout=60)
        if out.returncode:
            raise RuntimeError("openssl failed: " + out.stderr.strip()[-300:])
    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2", "-subj", "/CN=llm-eval canned pages CA",
            "-keyout", str(d / "ca.key"), "-out", str(d / "ca.pem"))
    openssl("req", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=llm-eval canned pages",
            "-keyout", str(d / "leaf.key"), "-out", str(d / "leaf.csr"))
    (d / "leaf.ext").write_text("subjectAltName=" + ",".join(f"DNS:{h}" for h in sorted(hosts)) + "\n")
    openssl("x509", "-req", "-in", str(d / "leaf.csr"), "-CA", str(d / "ca.pem"), "-CAkey", str(d / "ca.key"),
            "-CAcreateserial", "-days", "2", "-extfile", str(d / "leaf.ext"), "-out", str(d / "leaf.pem"))
    return d / "ca.pem", d / "leaf.pem", d / "leaf.key"


class CannedPages:
    def __init__(self, pages, cert_dir):
        """``pages``: url -> {"body", optional "status", "content_type"}."""
        self.pages = {self.key(url): page for url, page in pages.items()}
        self.hits, self.misses, self.harness = [], [], []
        hosts = {urlsplit(url).hostname for url in pages if url.startswith("https://")}
        self.tls = None
        if hosts:
            self.ca, cert, key = make_ca(cert_dir, hosts)
            self.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self.tls.load_cert_chain(cert, key)
            self.tls_hosts = hosts
        handler = type("Handler", (_Proxy,), {"owner": self})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def key(url):
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.hostname}{parts.path or '/'}" + (f"?{parts.query}" if parts.query else "")

    def lookup(self, url):
        page = self.pages.get(self.key(url))
        self._record(url, page)
        return page

    def _record(self, url, page=None):
        if page:
            self.hits.append(url)
        elif (urlsplit(url).hostname or "") in HARNESS_HOSTS:
            self.harness.append(url)
        else:
            self.misses.append(url)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class _Proxy(BaseHTTPRequestHandler):
    owner = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _serve(self, url):
        page = self.owner.lookup(url)
        body = (page["body"] if page else f"llm-eval: {url} is not in this task's canned pages.\n").encode()
        self.send_response(page.get("status", 200) if page else 404)
        self.send_header("Content-Type", page.get("content_type", "text/html; charset=utf-8") if page else "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("http://") or self.path.startswith("https://"):
            return self._serve(self.path)
        host = getattr(self, "tls_host", None)
        return self._serve(f"https://{host}{self.path}" if host else f"http://unknown{self.path}")

    do_HEAD = do_GET

    def do_CONNECT(self):
        host = self.path.rsplit(":", 1)[0]
        if not self.owner.tls or host not in self.owner.tls_hosts:
            self.owner._record(f"https://{host}/")
            self.send_error(403, "host not in canned pages")
            return
        self.send_response(200, "Connection Established")
        self.end_headers()
        try:
            tunnel = self.owner.tls.wrap_socket(self.connection, server_side=True)
        except (ssl.SSLError, OSError):
            return
        inner = type("Inner", (_Proxy,), {"owner": self.owner, "tls_host": host})
        try:
            inner(tunnel, self.client_address, self.server)
        except (OSError, ssl.SSLError):
            pass
        finally:
            try:
                tunnel.close()
            except OSError:
                pass
        self.close_connection = True


def sandbox_env(ca_inside):
    """Env that sends webfetch through the proxy while LLM calls stay direct to the relay."""
    proxy = f"http://{PROXY_INSIDE}"
    return {"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "http_proxy": proxy, "https_proxy": proxy,
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            "NODE_EXTRA_CA_CERTS": ca_inside}
