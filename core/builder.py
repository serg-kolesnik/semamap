"""UMAP-проекция, раскладка на сетку и нарезка тайлов (Deep Zoom)."""
import gc
import io
import json
import math
import os
import shutil
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import BG, map_dir
from .database import connect, meta_get, migrate


# ---------------- проекция ----------------
def project(E, a):
    import umap
    from sklearn.decomposition import PCA

    E = np.asarray(E, dtype=np.float32)  # PCA/UMAP на float16 работают плохо
    n, dim = E.shape
    print("1/2: Сжимаем размерность через PCA...")
    E_small = PCA(n_components=min(50, n, dim), random_state=42).fit_transform(E)

    print("2/2: Запускаем UMAP...")
    r = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(a.neighbors, n - 1)),
        min_dist=a.min_dist,
        metric="euclidean",
        low_memory=True,
        verbose=True,
    )
    return r.fit_transform(E_small)


def place_new(E, ids, old_ids, old_xy):
    """Кладёт новые фото рядом с ближайшим старым (без пересчёта UMAP)."""
    row_of_old = {int(i): k for k, i in enumerate(old_ids)}
    keep = np.array([int(i) in row_of_old for i in ids])
    xy = np.zeros((len(ids), 2), dtype=np.float32)
    xy_old = old_xy[[row_of_old[int(i)] for i in ids[keep]]]
    xy[keep] = xy_old
    if (~keep).any():
        E_old = E[keep].astype(np.float32)
        E_new = E[~keep].astype(np.float32)
        new_rows = np.nonzero(~keep)[0]
        rng = np.random.default_rng(0)
        for s in range(0, len(E_new), 2048):
            nearest = np.argmax(E_new[s:s + 2048] @ E_old.T, axis=1)
            base = xy_old[nearest]
            xy[new_rows[s:s + 2048]] = base + rng.normal(0, 0.05, base.shape)
    return xy


# ---------------- раскладка ----------------
def _free_in_ring(grid, cx, cy, r, W, H):
    for dx in range(-r, r + 1):
        for dy in ((-r, r) if abs(dx) != r else range(-r, r + 1)):
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < W - 1 and 1 <= ny < H - 1 and grid[ny, nx] == -1:
                return nx, ny
    return None


def layout(xy, W, H):
    n = len(xy)
    grid = np.full((H, W), -1, dtype=np.int64)

    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
    x_pad = (x_max - x_min) * 0.05 + 1e-9
    y_pad = (y_max - y_min) * 0.05 + 1e-9
    gx = ((xy[:, 0] - (x_min - x_pad)) / ((x_max + x_pad) - (x_min - x_pad)) * (W - 1)).astype(int)
    gy = ((xy[:, 1] - (y_min - y_pad)) / ((y_max + y_pad) - (y_min - y_pad)) * (H - 1)).astype(int)
    gx = np.clip(gx, 1, W - 2)
    gy = np.clip(gy, 1, H - 2)

    max_r = max(W, H)
    for i in range(n):
        cx, cy = int(gx[i]), int(gy[i])
        if grid[cy, cx] == -1:
            grid[cy, cx] = i
            continue
        # Ищем ближайшую свободную клетку без ограничения радиуса
        # (раньше при радиусе >19 фото затирало соседа и пропадало с карты).
        for r in range(1, max_r):
            hit = _free_in_ring(grid, cx, cy, r, W, H)
            if hit:
                grid[hit[1], hit[0]] = i
                break
        else:
            raise RuntimeError("Не хватило места в сетке для всех фото")
    return grid


# ---------------- тайлы ----------------
_G = {}


def _init_leaf(db_path, gid, cell, T, outdir, L, quality):
    _G.update(con=sqlite3.connect(db_path), gid=gid, cell=cell, T=T, out=Path(outdir), L=L, q=quality)


def _leaf_tile(t):
    tx, ty = t
    g = _G
    cell, T = g["cell"], g["T"]
    cpt = T // cell
    sub = g["gid"][ty * cpt:(ty + 1) * cpt, tx * cpt:(tx + 1) * cpt]
    canvas = Image.new("RGB", (sub.shape[1] * cell, sub.shape[0] * cell), BG)
    ids = [int(v) for v in sub.ravel() if v >= 0]
    if ids:
        d = {}
        for s in range(0, len(ids), 500):  # лимит SQLite на число параметров
            part = ids[s:s + 500]
            sql = "SELECT id, jpg FROM thumbs WHERE id IN (%s)" % ",".join("?" * len(part))
            d.update(g["con"].execute(sql, part).fetchall())
        for (cy, cx), v in np.ndenumerate(sub):
            if v >= 0:
                data = d.get(int(v))
                if data:
                    with Image.open(io.BytesIO(data)) as thumb:
                        canvas.paste(thumb, (cx * cell, cy * cell))
    canvas.save(g["out"] / str(g["L"]) / f"{tx}_{ty}.jpg", "JPEG", quality=g["q"])


def _parent_tile(t):
    tx, ty, k, T, dim_w, dim_h, outdir, quality = t
    out = Path(outdir)
    cw = min(2 * T, dim_w - 2 * tx * T)
    ch = min(2 * T, dim_h - 2 * ty * T)
    canvas = Image.new("RGB", (cw, ch), BG)
    for j in range(2):
        for i in range(2):
            p = out / str(k + 1) / f"{2 * tx + i}_{2 * ty + j}.jpg"
            if p.exists():
                with Image.open(p) as im:
                    canvas.paste(im, (i * T, j * T))
    small = canvas.resize((max(1, (cw + 1) // 2), max(1, (ch + 1) // 2)), Image.BOX)
    small.save(out / str(k) / f"{tx}_{ty}.jpg", "JPEG", quality=quality)


def build_pyramid(db_path, out_root, gid, cell, T, workers, quality=88):
    from tqdm import tqdm
    H, W = gid.shape
    PW, PH = W * cell, H * cell
    L = max(1, math.ceil(math.log2(max(PW, PH))))
    out = Path(out_root) / "map_files"
    shutil.rmtree(out, ignore_errors=True)
    for k in range(L + 1):
        (out / str(k)).mkdir(parents=True)
    nx, ny = math.ceil(PW / T), math.ceil(PH / T)
    tasks = [(tx, ty) for ty in range(ny) for tx in range(nx)]
    with ProcessPoolExecutor(workers, initializer=_init_leaf,
                             initargs=(str(db_path), gid, cell, T, str(out), L, quality)) as ex:
        list(tqdm(ex.map(_leaf_tile, tasks, chunksize=8), total=len(tasks), desc=f"Тайлы {L}"))
        for k in range(L - 1, -1, -1):
            dw = math.ceil(PW / 2 ** (L - k - 1))
            dh = math.ceil(PH / 2 ** (L - k - 1))
            pw, ph = math.ceil(PW / 2 ** (L - k)), math.ceil(PH / 2 ** (L - k))
            jobs = [(tx, ty, k, T, dw, dh, str(out), quality)
                    for ty in range(math.ceil(ph / T)) for tx in range(math.ceil(pw / T))]
            if len(jobs) > 64:
                list(tqdm(ex.map(_parent_tile, jobs, chunksize=8), total=len(jobs), desc=f"Тайлы {k}"))
            else:
                for j in jobs:
                    _parent_tile(j)


# ---------------- 3D: координаты и атлас ----------------
def build_coords3d(mdir, ids, xy, dates):
    """X = время, Y/Z = сходство (та же UMAP-проекция)."""
    if len(dates) >= 20:   # обрезаем выбросы (битые EXIF), чтобы ось времени не сжималась
        lo, hi = np.percentile(dates, [0.5, 99.5])
    else:
        lo, hi = dates.min(), dates.max()
    t = np.clip((dates - lo) / (hi - lo + 1e-9), 0, 1)
    c = xy - (xy.min(0) + xy.max(0)) / 2
    scale = 100.0 / max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
    coords = np.zeros((len(ids), 3), dtype=np.float32)
    coords[:, 0] = (t * 2 - 1) * 100
    coords[:, 1] = c[:, 0] * scale
    coords[:, 2] = c[:, 1] * scale
    np.savez(Path(mdir) / "coords3d.npz", ids=ids, coords=coords, dates=dates,
             range=np.array([lo, hi], dtype=np.float64))


def build_atlas(db_path, mdir, ids):
    """Склеивает миниатюры в одну текстуру. Порядок ячеек = порядок ids (= порядок точек в coords3d)."""
    n = len(ids)
    grid = math.ceil(math.sqrt(n))
    cell = max(8, min(64, 8192 // grid))          # текстура не больше 8192 px
    atlas = Image.new("RGB", (grid * cell, grid * cell), BG)
    con = sqlite3.connect(str(db_path))
    try:
        for s0 in range(0, n, 500):
            part = [int(i) for i in ids[s0:s0 + 500]]
            sql = "SELECT id, jpg FROM thumbs WHERE id IN (%s)" % ",".join("?" * len(part))
            d = dict(con.execute(sql, part).fetchall())
            for k, id_ in enumerate(part):
                jpg = d.get(id_)
                if not jpg:
                    continue
                idx = s0 + k
                with Image.open(io.BytesIO(jpg)) as im:
                    im = im.convert("RGB").resize((cell, cell), Image.LANCZOS)
                    atlas.paste(im, ((idx % grid) * cell, (idx // grid) * cell))
    finally:
        con.close()
    atlas.save(Path(mdir) / "atlas.jpg", "JPEG", quality=85)
    (Path(mdir) / "atlas_meta.json").write_text(json.dumps({"grid": grid, "cell": cell, "n": n}))
    print(f"Атлас: {grid * cell}×{grid * cell}, ячейка {cell}px, {n} фото")


# ---------------- команда build ----------------
def _ts(s, end=False):
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        sys.exit(f"Неверная дата '{s}', нужен формат ГГГГ-ММ-ДД")
    if end:
        d = d.replace(hour=23, minute=59, second=59)
    return d.timestamp()


def run_build(a):
    data = Path(a.data)
    db_path = data / "db.sqlite"
    if not db_path.exists():
        sys.exit("Сначала выполните index")
    con = connect(db_path)
    migrate(con)
    cell = int(meta_get(con, "cell") or 0)
    if not cell:
        sys.exit("Сначала index")
    T = a.tile or cell * max(2, 512 // cell)
    if T % cell:
        sys.exit("--tile не делится на cell")

    # Карта за период пишется в data/maps/<имя>, полная — в data/ (не затирают друг друга)
    name = getattr(a, "name", None)
    if not name and (a.date_from or a.date_to):
        name = f"{a.date_from or 'start'}_{a.date_to or 'now'}"
    mdir = map_dir(data, name)
    mdir.mkdir(parents=True, exist_ok=True)
    print(f"Папка карты: {mdir}")

    q = "SELECT id, emb, COALESCE(exif_date, mtime) FROM images WHERE bad=0 AND emb IS NOT NULL"
    params = []
    if a.date_from:
        q += " AND COALESCE(exif_date, mtime) >= ?"
        params.append(_ts(a.date_from))
    if a.date_to:
        q += " AND COALESCE(exif_date, mtime) <= ?"
        params.append(_ts(a.date_to, end=True))
    q += " ORDER BY id"

    print("Загрузка эмбеддингов из БД...")
    rows = con.execute(q, params).fetchall()
    con.close()
    n = len(rows)
    if a.date_from or a.date_to:
        print(f"Фото в диапазоне {a.date_from or '…'} — {a.date_to or '…'}: {n}")
    if n < 4:
        sys.exit("Мало картинок")

    print(f"Собрано {n} векторов. Распаковка в память...")
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    dates = np.array([r[2] for r in rows], dtype=np.float64)
    E = np.stack([np.frombuffer(r[1], dtype=np.float16) for r in rows])
    del rows

    xy_file = mdir / "xy.npz"
    xy = None
    if xy_file.exists() and not a.reproject:
        with np.load(xy_file) as c:
            old_ids, old_xy = c["ids"], c["xy"]
        if np.array_equal(old_ids, ids):
            xy = old_xy
            print("Беру сохранённую UMAP (пропускаем пересчёт).")
        elif np.isin(ids, old_ids).any():
            print("Найдены изменения в коллекции — раскладываем новые фото по соседям (быстрый режим)...")
            xy = place_new(E, ids, old_ids, old_xy)
            np.savez(xy_file, ids=ids, xy=xy)
    if xy is None:
        xy = project(E, a)
        np.savez(xy_file, ids=ids, xy=xy)

    x_rng = max(float(xy[:, 0].max() - xy[:, 0].min()), 1e-6)
    y_rng = max(float(xy[:, 1].max() - xy[:, 1].min()), 1e-6)
    aspect = x_rng / y_rng
    area = n * 7.5
    H = max(10, int(math.sqrt(area / aspect)))
    W = max(10, int(math.sqrt(area * aspect)))

    print(f"Органическая раскладка {n} по {W}×{H}…")
    gc.collect()
    grid = layout(xy, W, H)
    np.savez(mdir / "layout.npz", ids=ids, grid=grid.astype(np.int32), cell=cell, tile=T)
    np.save(mdir / "emb.npy", E)
    gid = np.where(grid >= 0, ids[np.clip(grid, 0, None)], -1)

    workers = a.workers if a.workers is not None else (os.cpu_count() or 2)
    print(f"Запуск нарезки тайлов в {workers} процессов...")
    build_pyramid(db_path, mdir, gid, cell, T, workers)

    if not getattr(a, "no_3d", False):
        print("3D: координаты (X=время, Y/Z=сходство) и атлас текстур...")
        build_coords3d(mdir, ids, xy, dates)
        build_atlas(db_path, mdir, ids)

    cmd = f"python main.py serve" + (f" --map {mdir.name}" if name else "")
    print(f"Готово! Запуск: {cmd}")
