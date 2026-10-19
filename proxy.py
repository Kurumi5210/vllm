import hmac, itertools, json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import httpx
UPSTREAM = "https://api-gateway.glm.ai"
UPSTREAM_KEY = os.environ.get("UPSTREAM_API_KEY", "").strip()
PROXY_KEY = os.environ.get("PROXY_API_KEY", "").strip()
HOST, PORT = "127.0.0.1", 9000
REQUIRE_AUTH = True
ALLOWED = ("/v1", "/chat", "/responses", "/beta", "/anthropic")
MAX_BODY = 100 * 1024 * 1024
PASS = {"accept", "content-type", "anthropic-version", "anthropic-beta"}
HOP = {"connection","keep-alive","proxy-authenticate","proxy-authorization",
       "te","trailer","transfer-encoding","upgrade"}
CLIENT = httpx.Client(
    timeout=httpx.Timeout(connect=15, read=600, write=60, pool=30),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=50,
                        keepalive_expiry=300),
    follow_redirects=False, trust_env=True,
)
REQ_ID = itertools.count(1)
def get_usage(obj):
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("usage"), dict):
        return obj["usage"]
    for k in ("message", "response"):
        x = obj.get(k)
        if isinstance(x, dict) and isinstance(x.get("usage"), dict):
            return x["usage"]
    return None
def log_usage(i, u):
    if not isinstance(u, dict):
        return
    prompt = u.get("prompt_tokens", u.get("input_tokens"))
    output = u.get("completion_tokens", u.get("output_tokens"))
    hit = miss = write = None
    src = "generic"
    if "prompt_cache_hit_tokens" in u or "prompt_cache_miss_tokens" in u:
        src = "deepseek"
        hit = u.get("prompt_cache_hit_tokens", 0)
        miss = u.get("prompt_cache_miss_tokens", 0)
    elif isinstance(u.get("prompt_tokens_details"), dict):
        d = u["prompt_tokens_details"]
        if "cached_tokens" in d:
            src = "openai"
            hit = d.get("cached_tokens", 0)
            miss = max(prompt - hit, 0) if isinstance(prompt, int) else None
    elif "cache_read_input_tokens" in u or "cache_creation_input_tokens" in u:
        src = "anthropic"
        raw = u.get("input_tokens", 0) or 0
        hit = u.get("cache_read_input_tokens", 0) or 0
        write = u.get("cache_creation_input_tokens", 0) or 0
        miss = raw + write
        prompt = raw + hit + write
    parts = [f"[USAGE #{i}] source={src}", f"prompt={prompt}",
             f"completion={output}"]
    if hit is not None: parts.append(f"cache_hit={hit}")
    if miss is not None: parts.append(f"cache_miss={miss}")
    if write is not None: parts.append(f"cache_write={write}")
    if isinstance(hit, int) and isinstance(miss, int) and hit + miss:
        parts.append(f"hit_rate={hit/(hit+miss)*100:.2f}%")
    print(" ".join(parts), flush=True)
class SSEUsage:
    def __init__(self):
        self.buf = bytearray()
        self.usage = None
    def feed(self, chunk):
        self.buf.extend(chunk)
        while b"\n" in self.buf:
            line, _, rest = self.buf.partition(b"\n")
            self.buf = bytearray(rest)
            line = line.strip()
            if line.startswith(b"data:"): line = line[5:].strip()
            if not line.startswith(b"{"): continue
            try:
                u = get_usage(json.loads(line))
                if u: self.usage = u
            except Exception:
                pass
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LLMProxy/6.2"
    wbufsize = 0
    def do_GET(self):
        if urlsplit(self.path).path != "/health":
            return self.send_json(404, {"error": "Not found"})
        self.send_json(200, {"status":"ok","version":"6.2",
                            "upstream":UPSTREAM,"proxy_auth":REQUIRE_AUTH})
    def do_POST(self):
        i = next(REQ_ID)
        start = time.perf_counter()
        p = urlsplit(self.path)
        path = p.path
        ip = self.headers.get("CF-Connecting-IP") or self.client_address[0]
        print(f"\n[REQUEST #{i}] POST {self.path} from={ip}", flush=True)
        if not any(path == x or path.startswith(x + "/") for x in ALLOWED):
            return self.send_json(404, {"error":"Path not allowed"})
        if REQUIRE_AUTH and not self.auth_ok():
            return self.send_json(401, {"error":"Invalid proxy API key"})
        try:
            body = self.read_body()
            payload = json.loads(body) if body else {}
        except Exception as e:
            return self.send_json(400, {"error":str(e)})
        headers = {k:v for k,v in self.headers.items() if k.lower() in PASS}
        headers.setdefault("Content-Type", "application/json")
        headers["Accept-Encoding"] = "identity"
        headers["Authorization"] = f"Bearer {UPSTREAM_KEY}"
        url = UPSTREAM.rstrip("/") + "/" + path.lstrip("/")
        if p.query: url += "?" + p.query
        started = False
        first = None
        sent = chunks = 0
        try:
            with CLIENT.stream("POST", url, headers=headers, content=body) as r:
                up = time.perf_counter()
                ctype = r.headers.get("content-type", "").lower()
                stream = payload.get("stream") is True or "text/event-stream" in ctype
                if stream:
                    started = True
                    self.send_response(r.status_code)
                    self.copy_headers(r.headers)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    inspect = SSEUsage()
                    for chunk in r.iter_raw():
                        if not chunk: continue
                        first = first or time.perf_counter()
                        sent += len(chunk); chunks += 1
                        inspect.feed(chunk)
                        self.wfile.write(f"{len(chunk):X}\r\n".encode())
                        self.wfile.write(chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    log_usage(i, inspect.usage)
                else:
                    raw = r.read()
                    first = time.perf_counter()
                    sent, chunks = len(raw), 1
                    try: log_usage(i, get_usage(json.loads(raw)))
                    except Exception: pass
                    started = True
                    self.send_response(r.status_code)
                    self.copy_headers(r.headers)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    self.wfile.flush()
                end = time.perf_counter()
                ttfb = (first or up) - start
                print(
                    f"[SUMMARY #{i}] status={r.status_code} path={path} "
                    f"upstream={(up-start)*1000:.0f}ms "
                    f"ttfb={ttfb*1000:.0f}ms total={(end-start)*1000:.0f}ms "
                    f"bytes={sent} chunks={chunks}", flush=True
                )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            print(f"[CLIENT-DISCONNECTED #{i}]", flush=True)
        except httpx.RequestError as e:
            self.close_connection = True
            print(f"[UPSTREAM-ERROR #{i}] {type(e).__name__}: {e}", flush=True)
            if not started: self.send_json(502, {"error":"Upstream connection failed"})
        except Exception as e:
            self.close_connection = True
            print(f"[PROXY-ERROR #{i}] {type(e).__name__}: {e}", flush=True)
            if not started: self.send_json(500, {"error":"Internal proxy error"})
    def auth_ok(self):
        a = self.headers.get("Authorization", "")
        key = a[7:].strip() if a.lower().startswith("bearer ") \
              else self.headers.get("x-api-key", "").strip()
        return bool(key and PROXY_KEY and hmac.compare_digest(key, PROXY_KEY))
    def read_body(self):
        length = self.headers.get("Content-Length")
        if length is not None:
            n = int(length)
            if not 0 <= n <= MAX_BODY: raise ValueError("Invalid body size")
            data = self.rfile.read(n) if n else b""
            if len(data) != n: raise ConnectionError("Incomplete body")
            return data
        if "chunked" not in self.headers.get("Transfer-Encoding","").lower():
            return b""
        body = bytearray()
        while True:
            line = self.rfile.readline()
            if not line: raise ConnectionError("Client disconnected")
            size = int(line.strip().split(b";",1)[0], 16)
            if size == 0:
                while self.rfile.readline() not in (b"",b"\r\n",b"\n"): pass
                return bytes(body)
            chunk = self.rfile.read(size)
            if len(chunk) != size: raise ConnectionError("Incomplete chunk")
            body.extend(chunk)
            if len(body) > MAX_BODY: raise ValueError("Body too large")
            if self.rfile.read(2) != b"\r\n": raise ValueError("Bad chunk framing")
    def copy_headers(self, headers):
        skip = HOP | {"content-length","content-encoding","server","date"}
        for k,v in headers.items():
            if k.lower() not in skip: self.send_header(k,v)
    def send_json(self, status, obj):
        raw = json.dumps(obj, ensure_ascii=False, separators=(",",":")).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(raw)))
        self.end_headers()
        self.wfile.write(raw); self.wfile.flush()
    def log_message(self, *_):
        pass
def main():
    if not UPSTREAM_KEY: raise SystemExit("Missing UPSTREAM_API_KEY")
    if REQUIRE_AUTH and not PROXY_KEY: raise SystemExit("Missing PROXY_API_KEY")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    server.request_queue_size = 128
    print(f"LLM Proxy v6.2 | http://{HOST}:{PORT} -> {UPSTREAM} | auth={REQUIRE_AUTH}",
          flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        CLIENT.close()
if __name__ == "__main__":
    main()
