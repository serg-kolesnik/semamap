import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import BG, DETAIL_SIZE, EXTS
from .database import SCHEMA, connect, meta_get, meta_set, migrate

Image.MAX_IMAGE_PIXELS = None

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass


def load_rgb(path, hint=1024):
    with Image.open(path) as im:
        try:
            im.draft("RGB", (hint, hint))
        except Exception:
            pass
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            return bg
        return im.convert("RGB")


def thumb_jpeg(im, size, quality=88):
    im = ImageOps.pad(im, (size, size), Image.LANCZOS, color=BG)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality)
    return b.getvalue()


# ---------- дата съёмки ----------
def _parse_exif_dt(v):
    if not v:
        return None
    if isinstance(v, bytes):
        v = v.decode("ascii", "ignore")
    v = str(v).replace("\x00", "").strip()[:19]
    try:
        dt = datetime.strptime(v, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None          # в т.ч. "0000:00:00 00:00:00"
    return dt.timestamp() if dt.year >= 1990 else None


def get_exif_date(path):
    """Дата съёмки из EXIF или None. DateTimeOriginal/Digitized лежат в Exif-подкаталоге (0x8769),
    а не в корне, поэтому просто exif.get(36867) их не находит."""
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            try:
                sub = ex.get_ifd(0x8769)
            except Exception:
                sub = {}
            for v in (sub.get(36867), sub.get(36868), ex.get(306)):
                ts = _parse_exif_dt(v)
                if ts:
                    return ts
    except Exception:
        pass
    return None


class PathDataset:
    def __init__(self, paths, preprocess, cell):
        self.paths, self.preprocess, self.cell = paths, preprocess, cell

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            p = self.paths[i]
            im = load_rgb(p)
            return (i, self.preprocess(im), thumb_jpeg(im, self.cell),
                    thumb_jpeg(im, DETAIL_SIZE, 85), get_exif_date(p))
        except Exception:
            return i, None, None, None, None


def collate(batch):
    import torch
    ok = [b for b in batch if b[1] is not None]
    bad = [b[0] for b in batch if b[1] is None]
    xs = torch.stack([b[1] for b in ok]) if ok else None
    return ([b[0] for b in ok], xs, [b[2] for b in ok], [b[3] for b in ok],
            [b[4] for b in ok], bad)


def scan_dirs(roots):
    found = {}
    for r in roots:
        for dp, dn, fn in os.walk(r):
            dn[:] = [d for d in dn if not d.startswith(".")]
            for f in fn:
                if f.startswith(".") or os.path.splitext(f)[1].lower() not in EXTS:
                    continue
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                found[p] = (st.st_mtime, st.st_size)
    return found


# ---------- дозаполнение старой базы (без пересчёта CLIP) ----------
def _backfill_one(job):
    path, need_thumb = job
    jpg = None
    if need_thumb:
        try:
            jpg = thumb_jpeg(load_rgb(path), DETAIL_SIZE, 85)
        except Exception:
            pass
    return get_exif_date(path), jpg


def backfill(con, workers=None):
    rows = con.execute(
        "SELECT i.id, i.path, i.mtime, i.exif_date IS NULL, t.id IS NULL "
        "FROM images i LEFT JOIN thumbs512 t ON t.id=i.id "
        "WHERE i.bad=0 AND (i.exif_date IS NULL OR t.id IS NULL)").fetchall()
    if not rows:
        return
    print(f"Дозаполняю даты EXIF и HD-превью у {len(rows)} фото…")
    workers = workers if workers is not None else min(8, os.cpu_count() or 2)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **k: it
    jobs = [(r[1], bool(r[4])) for r in rows]
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for r, (exif, jpg) in zip(rows, tqdm(ex.map(_backfill_one, jobs, chunksize=16),
                                             total=len(rows), desc="Дозаполнение")):
            id_, _, mtime, need_date, _ = r
            if need_date:
                con.execute("UPDATE images SET exif_date=? WHERE id=?", (exif or mtime, id_))
            if jpg:
                con.execute("INSERT OR REPLACE INTO thumbs512(id,jpg) VALUES(?,?)", (id_, jpg))
    con.commit()


def run_index(a):
    import torch
    import open_clip
    from tqdm import tqdm

    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    con = connect(data / "db.sqlite")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    migrate(con)

    want = f"{a.model}|{a.pretrained}|{a.cell}"
    prev = meta_get(con, "signature")
    if prev and prev != want:
        if not a.force:
            sys.exit("База построена с другими настройками. Используйте --force.")
        for t in ("images", "thumbs", "thumbs512"):
            con.execute(f"DELETE FROM {t}")
    meta_set(con, "signature", want)
    meta_set(con, "model", a.model)
    meta_set(con, "pretrained", a.pretrained)
    meta_set(con, "cell", a.cell)
    con.commit()

    roots = [os.path.abspath(r) for r in a.dirs]
    print("Сканирую папки…")
    found = scan_dirs(roots)

    existing = {p: (m, s, i) for i, p, m, s in con.execute("SELECT id,path,mtime,size FROM images")}
    gone = []
    for p, (m, s, i) in existing.items():
        if p in found:
            if found[p] != (m, s):
                gone.append((i, p))
        elif any(p.startswith(r + os.sep) for r in roots):
            gone.append((i, p))
    for i, p in gone:
        for t in ("images", "thumbs", "thumbs512"):
            con.execute(f"DELETE FROM {t} WHERE id=?", (i,))
        existing.pop(p, None)
    con.commit()

    backfill(con, a.workers)   # старые записи: даты EXIF + HD-превью, без CLIP

    todo = [p for p in found if p not in existing]
    print(f"Найдено: {len(found)}, новых: {len(todo)}, удалённых/изменённых: {len(gone)}")
    if not todo:
        return

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print(f"Модель {a.model}/{a.pretrained} на {dev}")
    model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
    model = model.to(dev).eval()

    workers = a.workers if a.workers is not None else min(8, os.cpu_count() or 2)
    loader = torch.utils.data.DataLoader(
        PathDataset(todo, preprocess, a.cell), batch_size=a.batch,
        num_workers=workers, collate_fn=collate, persistent_workers=False)
    n_ok = 0
    with torch.no_grad():
        for step, (idxs, xs, thumbs, big, dates, bad) in enumerate(
                tqdm(loader, desc="Индексация", unit="batch")):
            if idxs:
                x = xs.to(dev)
                if dev == "cuda":
                    with torch.autocast("cuda"):
                        f = model.encode_image(x)
                else:
                    f = model.encode_image(x)
                f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype(np.float16)
                meta_set(con, "dim", f.shape[1])
                for k, i in enumerate(idxs):
                    p = todo[i]
                    m, s = found[p]
                    cur = con.execute(
                        "INSERT INTO images(path,mtime,exif_date,size,emb) VALUES(?,?,?,?,?)",
                        (p, m, dates[k] or m, s, f[k].tobytes()))
                    con.execute("INSERT INTO thumbs(id,jpg) VALUES(?,?)", (cur.lastrowid, thumbs[k]))
                    con.execute("INSERT INTO thumbs512(id,jpg) VALUES(?,?)", (cur.lastrowid, big[k]))
                n_ok += len(idxs)
            for i in bad:
                p = todo[i]
                m, s = found[p]
                con.execute("INSERT INTO images(path,mtime,exif_date,size,bad) VALUES(?,?,?,?,1)",
                            (p, m, m, s))
            if step % 20 == 0:
                con.commit()
    con.commit()
    print(f"Готово: {n_ok}. Дальше: python main.py build")
