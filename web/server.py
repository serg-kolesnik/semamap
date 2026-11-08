"""HTTP-сервер и API."""
import base64
import hmac
import io
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from core.config import CYR, is_nsfw_text, is_nsfw_vocab, load_vocab, map_dir
from core.database import connect, meta_get
from core.indexer import load_rgb
from core.tagging import (build_labels, encode_vocab, precompute_image_tags,
                          precompute_nsfw_tags)

STATIC = Path(__file__).parent / "static"
CTYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
          ".js": "application/javascript; charset=utf-8", ".jpg": "image/jpeg"}


def topk(scores, k):
    k = max(1, min(k, len(scores)))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def load_points3d(mdir):
    """Готовый JSON для 3D-вида или None, если карта собрана без 3D."""
    mdir = Path(mdir)
    if not all((mdir / f).exists() for f in ("coords3d.npz", "atlas.jpg", "atlas_meta.json")):
        return None
    with np.load(mdir / "coords3d.npz") as c:
        co = c["coords"]
        return json.dumps({
            "ids": c["ids"].tolist(),
            "x": np.round(co[:, 0], 2).tolist(), "y": np.round(co[:, 1], 2).tolist(),
            "z": np.round(co[:, 2], 2).tolist(),
            "date": np.round(c["dates"]).tolist(), "range": c["range"].tolist(),
        }).encode()


def safe_file(base, rel):
    """Файл внутри base или None (защита от ../ в пути)."""
    base = Path(base).resolve()
    f = (base / rel).resolve()
    try:
        f.relative_to(base)
    except ValueError:
        return None
    return f if f.is_file() else None


def parse_path_map(cli_values, env_value):
    """Пары «путь в базе=путь в контейнере», напр. C:\\photos=/photos (флаг --path-map или SEMAMAP_PATH_MAP через ;)."""
    raw = list(cli_values or []) + [x for x in (env_value or "").split(";") if x.strip()]
    pairs = []
    for item in raw:
        src, sep, dst = item.partition("=")
        if sep and src.strip() and dst.strip():
            pairs.append((src.strip(), dst.strip()))
    return pairs


def remap(p, pairs):
    """Подменяет префикс пути из базы (Windows) на путь внутри контейнера."""
    if not p or not pairs:
        return p
    n = p.replace("\\", "/")
    for src, dst in pairs:
        s_ = src.replace("\\", "/").rstrip("/")
        if n.lower() == s_.lower() or n.lower().startswith(s_.lower() + "/"):
            return dst.rstrip("/") + n[len(s_):]
    return p


class Handler(BaseHTTPRequestHandler):
    S = None

    # ---------- утилиты ----------
    def send_bytes(self, b, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj).encode(), "application/json", code)

    def send_static(self, f):
        extra = {"Cache-Control": "no-store"} if f.suffix in (".html", ".js", ".css") else None
        self.send_bytes(f.read_bytes(), CTYPES.get(f.suffix.lower(), "application/octet-stream"), extra=extra)

    def rows_to_items(self, rows, scores):
        S = self.S
        return [{"id": int(S.ids[r]), "x": int(S.pos[r, 0]), "y": int(S.pos[r, 1]),
                 "s": round(float(scores[r]), 3)} for r in rows]

    def path_of(self, id_):
        con = sqlite3.connect(self.S.db)
        try:
            r = con.execute("SELECT path FROM images WHERE id=?", (id_,)).fetchone()
        finally:
            con.close()
        return remap(r[0], self.S.path_map) if r else None

    def send_original(self, id_):
        p = self.path_of(id_)
        if not p or not os.path.isfile(p):
            # оригинала не видно (например, база с путями Windows, а сервер в контейнере) —
            # отдаём превью 512 px, чтобы «Открыть полностью» не было мёртвым
            return self.api_thumb512({"id": str(id_)})
        ext = os.path.splitext(p)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            with open(p, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(p)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                shutil.copyfileobj(f, self.wfile)
        else:
            b = io.BytesIO()
            load_rgb(p, hint=100000).save(b, "JPEG", quality=92)
            self.send_bytes(b.getvalue(), "image/jpeg")

    # ---------- авторизация ----------
    def authorized(self):
        S = self.S
        if not (S.auth_user and S.auth_pass):
            return True
        try:
            scheme, cred = self.headers.get("Authorization", "").split(" ", 1)
            if scheme.lower() != "basic":
                return False
            user, pwd = base64.b64decode(cred).decode("utf-8").split(":", 1)
        except Exception:
            return False
        return (hmac.compare_digest(user.encode(), S.auth_user.encode())
                and hmac.compare_digest(pwd.encode(), S.auth_pass.encode()))

    def request_auth(self):
        self.send_bytes(b"<html><body><h1>401 Unauthorized</h1></body></html>", "text/html", 401,
                        {"WWW-Authenticate": 'Basic realm="Semamap"'})

    # ---------- маршруты ----------
    def do_GET(self):
        if not self.authorized():
            return self.request_auth()
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            self.route(u.path, q)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (KeyError, ValueError) as e:
            self.safe_json({"error": f"Неверный запрос: {e}"}, 400)
        except Exception as e:
            self.safe_json({"error": str(e)}, 500)

    def safe_json(self, obj, code):
        try:
            self.send_json(obj, code)
        except Exception:
            pass

    def route(self, p, q):
        S = self.S
        if p in ("/", "/index.html"):
            self.send_static(STATIC / "index.html")
        elif p == "/3d":
            self.send_static(STATIC / "3d.html")
        elif p.startswith("/static/"):
            f = safe_file(STATIC, unquote(p[len("/static/"):]))
            self.send_static(f) if f else self.send_error(404)
        elif p.startswith("/map_files/"):
            f = safe_file(Path(S.data_dir) / "map_files", unquote(p[len("/map_files/"):]))
            self.send_static(f) if f else self.send_error(404)
        elif p == "/api/info":
            self.send_json({
                "cell": S.cell, "w": S.W, "h": S.H, "n": len(S.ids),
                "search": S.model is not None, "tags": S.V is not None,
                "labels": bool(S.labels), "translate": S.tr is not None,
                "nsfw": S.nsfw_signal is not None,
                "has3d": S.pts3d is not None,
                "tileSource": {"Image": {
                    "xmlns": "http://schemas.microsoft.com/deepzoom/2008",
                    "Url": "/map_files/", "Format": "jpg", "Overlap": "0",
                    "TileSize": S.tile_size,
                    "Size": {"Width": S.W * S.cell, "Height": S.H * S.cell}}},
            })
        elif p == "/api/cell":
            x, y = int(q["x"]), int(q["y"])
            if not (0 <= x < S.W and 0 <= y < S.H) or S.grid[y, x] < 0:
                return self.send_json({"id": -1})
            id_ = int(S.ids[S.grid[y, x]])
            self.send_json({"id": id_, "x": x, "y": y, "path": self.path_of(id_)})
        elif p == "/api/search":
            self.api_search(q)
        elif p == "/api/tags":
            self.api_tags(q)
        elif p == "/api/thumb512":
            self.api_thumb512(q)
        elif p == "/api/visible":
            self.api_visible(q)
        elif p == "/api/points3d":
            if S.pts3d is None:
                return self.send_json({"error": "3D не построено. Выполните build."}, 404)
            self.send_bytes(S.pts3d, "application/json")
        elif p == "/api/atlas_meta":
            f = safe_file(S.data_dir, "atlas_meta.json")
            self.send_static(f) if f else self.send_json({"error": "нет атласа"}, 404)
        elif p == "/api/atlas.jpg":
            f = safe_file(S.data_dir, "atlas.jpg")
            self.send_static(f) if f else self.send_error(404)
        elif p == "/api/labels":
            self.send_json({"levels": S.labels})
        elif p == "/api/similar":
            r = int(np.searchsorted(S.ids, int(q["id"])))
            if r >= len(S.ids) or S.ids[r] != int(q["id"]):
                return self.send_json([])
            sc = S.E @ S.E[r]
            self.send_json(self.rows_to_items(topk(sc, min(int(q.get("k", 200)), 2000)), sc))
        elif p.startswith("/file/"):
            self.send_original(int(p[6:]))
        else:
            self.send_error(404)

    def api_thumb512(self, q):
        id_ = int(q["id"])
        con = sqlite3.connect(self.S.db)
        try:
            r = None
            try:
                r = con.execute("SELECT jpg FROM thumbs512 WHERE id=?", (id_,)).fetchone()
            except sqlite3.OperationalError:      # старая база без thumbs512
                pass
            if not r:
                r = con.execute("SELECT jpg FROM thumbs WHERE id=?", (id_,)).fetchone()
        finally:
            con.close()
        if not r:
            return self.send_error(404)
        # id никогда не переиспользуется (AUTOINCREMENT), поэтому кэшируем надолго
        self.send_bytes(r[0], "image/jpeg", extra={"Cache-Control": "public, max-age=86400"})

    def api_visible(self, q):
        """Фото в видимой области (в клетках карты) — для подгрузки HD при приближении."""
        S = self.S
        x1, y1, x2, y2 = (float(q[k]) for k in ("x1", "y1", "x2", "y2"))
        mask = ((S.pos[:, 0] >= x1) & (S.pos[:, 0] <= x2) &
                (S.pos[:, 1] >= y1) & (S.pos[:, 1] <= y2))
        rows = np.nonzero(mask)[0]
        cap = max(1, min(int(q.get("k", 150)), 400))
        if len(rows) > cap:                      # оставляем ближайшие к центру экрана
            d = (S.pos[rows, 0] - (x1 + x2) / 2) ** 2 + (S.pos[rows, 1] - (y1 + y2) / 2) ** 2
            rows = rows[np.argsort(d)[:cap]]
        self.send_json([{"id": int(S.ids[r]), "x": int(S.pos[r, 0]), "y": int(S.pos[r, 1])}
                        for r in rows])

    def api_search(self, q):
        S = self.S
        if S.model is None:
            return self.send_json({"error": "Поиск отключён"}, 400)
        import torch
        text = q["q"].strip()
        if not text:
            return self.send_json({"items": [], "q_en": ""})
        en = q.get("en")
        with S.lock:
            if en is None and S.tr is not None and CYR.search(text):
                en = S.tr(text)
            query = en or text
            t_list = []
            with torch.no_grad():
                for tmpl in (f"a photo of {query}", f"a picture of {query}", query):
                    t_val = S.model.encode_text(S.tok([tmpl]))
                    t_list.append(torch.nn.functional.normalize(t_val.float(), dim=-1)[0])
            t = torch.stack(t_list).mean(dim=0).numpy()
        sc = S.E @ t

        # фильтр по видимой области (координаты в клетках)
        if all(k in q for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (float(q[k]) for k in ("x1", "y1", "x2", "y2"))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = max(x2 - x1, 200), max(y2 - y1, 200)
            x1, x2, y1, y2 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
            mask = ((S.pos[:, 0] >= x1) & (S.pos[:, 0] <= x2) &
                    (S.pos[:, 1] >= y1) & (S.pos[:, 1] <= y2))
            if mask.sum() < 10:
                mask = np.ones(len(S.ids), dtype=bool)
            sc = np.where(mask, sc, -np.inf)

        if S.nsfw_signal is not None:
            sc = sc + (0.3 if is_nsfw_text(query) else -0.1) * S.nsfw_signal

        valid = np.where(np.isfinite(sc))[0]
        q_en = en if en and en != text else ""
        if len(valid) == 0:
            return self.send_json({"items": [], "q_en": q_en})
        k = max(1, min(int(q.get("k", 200)), 2000))
        rows = valid[topk(sc[valid], k)]
        self.send_json({"items": self.rows_to_items(rows, sc), "q_en": q_en})

    def api_tags(self, q):
        S = self.S
        if S.V is None or not S.vocab:
            return self.send_json({"nsfw": False, "tags": []})
        id_ = int(q["id"])
        r = int(np.searchsorted(S.ids, id_))
        if r >= len(S.ids) or S.ids[r] != id_:
            return self.send_json({"nsfw": False, "tags": []})
        signal = float(S.nsfw_signal[r]) if S.nsfw_signal is not None else 0.0
        is_nsfw = signal > 0.05
        tags = []
        for i in (S.per_tags[r] if S.per_tags else []):
            tags.append({"ru": S.vocab[i][1], "en": S.vocab[i][0], "p": round(float(S.E[r] @ S.V[i]), 3)})
        if is_nsfw and S.per_nsfw_tags:
            for i in S.per_nsfw_tags[r]:
                tags.append({"ru": S.vocab[i][1], "en": S.vocab[i][0],
                             "p": round(float(S.E[r] @ S.V[i]), 3), "nsfw": True})
        tags.sort(key=lambda x: -x["p"])
        self.send_json({"nsfw": is_nsfw, "score": round(signal, 3), "tags": tags})


def run_serve(a):
    if bool(a.auth_user) != bool(a.auth_pass):
        sys.exit("Укажите и --auth-user, и --auth-pass (или ни одного).")
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.auth_user:
        print("ВНИМАНИЕ: сервер доступен из сети без пароля. Задайте --auth-user/--auth-pass.")

    data = Path(a.data).resolve()               # тут db.sqlite
    mdir = map_dir(data, getattr(a, "map", None))  # тут карта (layout, тайлы, эмбеддинги, 3D)
    if not (mdir / "layout.npz").exists() or not (mdir / "map_files").exists():
        sys.exit(f"Карта не найдена в {mdir}. Сначала выполните build.")
    print(f"Карта: {mdir}")

    lay = np.load(mdir / "layout.npz")
    ids, grid = lay["ids"], lay["grid"]
    H, W = grid.shape
    tile_size = int(lay["tile"])
    ys, xs = np.nonzero(grid >= 0)
    pos = np.zeros((len(ids), 2), dtype=np.int32)
    pos[grid[ys, xs]] = np.stack([xs, ys], 1)

    model = tok = None
    if not a.no_search:
        import open_clip
        con = connect(data / "db.sqlite")
        name, pre = meta_get(con, "model"), meta_get(con, "pretrained")
        con.close()
        print(f"Модель {name}/{pre}…")
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pre)
        model.eval()
        tok = open_clip.get_tokenizer(name)

    vocab = V = tr = None
    nsfw_idx = None
    nsfw_signal = nsfw_mask = None
    per_tags, per_nsfw_tags = [], []
    E = np.load(mdir / "emb.npy").astype(np.float32)

    if model is not None:
        vocab = load_vocab(a.vocab) or None
        if vocab:
            print(f"Словарь: {len(vocab)} понятий…")
            V = encode_vocab(model, tok, vocab)
            nsfw_mask = np.array([is_nsfw_vocab(en) for en, _ in vocab])
            nsfw_idx = np.nonzero(nsfw_mask)[0].tolist()
            sfw_idx = np.nonzero(~nsfw_mask)[0].tolist()
            print(f"NSFW-тегов: {len(nsfw_idx)}, SFW: {len(sfw_idx)}")
            if nsfw_idx and sfw_idx:
                nc = V[nsfw_idx].mean(0)
                nc /= np.linalg.norm(nc) + 1e-8
                sc = V[sfw_idx].mean(0)
                sc /= np.linalg.norm(sc) + 1e-8
                nsfw_signal = E @ (nc - sc)
                pos_count = int((nsfw_signal > 0.05).sum())
                print(f"NSFW-сигнал: mean={nsfw_signal.mean():.3f}, "
                      f"сработал на {pos_count}/{len(E)} ({100 * pos_count / max(1, len(E)):.1f}%)")
            print("Предрасчёт тегов…")
            per_tags = precompute_image_tags(E, V, nsfw_mask, top_n=12, min_cos=0.15)
            per_nsfw_tags = precompute_nsfw_tags(E, V, nsfw_idx, top_n=5, min_cos=0.35)

        if not a.no_translate and not re.search("xlm|multiling", name or "", re.I):
            try:
                print("Переводчик ru→en…")
                from core.translator import Translator
                tr = Translator()
            except Exception as e:
                print(f"Переводчик отключен ({e})")

    Handler.S = SimpleNamespace(
        ids=ids, grid=grid, pos=pos, W=W, H=H, cell=int(lay["cell"]), tile_size=tile_size,
        E=E, db=str(data / "db.sqlite"),
        model=model, tok=tok, lock=threading.Lock(),
        vocab=vocab, V=V, tr=tr, labels=[],
        data_dir=str(mdir), pts3d=load_points3d(mdir),
        path_map=parse_path_map(getattr(a, "path_map", None), os.environ.get("SEMAMAP_PATH_MAP")),
        auth_user=a.auth_user, auth_pass=a.auth_pass,
        nsfw_mask=nsfw_mask, nsfw_idx=nsfw_idx, nsfw_signal=nsfw_signal,
        per_tags=per_tags, per_nsfw_tags=per_nsfw_tags,
    )

    if V is not None and vocab:
        print("Подписи районов…")
        Handler.S.labels = build_labels(Handler.S)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}"
    print(f"Сервер запущен: {url}")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
