#!/usr/bin/env python3
"""semamap — карта фотоколлекции.

    python main.py index /путь/к/фото
    python main.py build
    python main.py serve
    python main.py build --date-from 2023-01-01 --date-to 2023-12-31   # карта за период
    python main.py serve --map 2023-01-01_2023-12-31
"""
import argparse
import os


def _index(a):
    from core.indexer import run_index
    run_index(a)


def _build(a):
    from core.builder import run_build
    run_build(a)


def _serve(a):
    from web.server import run_serve
    run_serve(a)


def main():
    ap = argparse.ArgumentParser(description="Semamap")
    ap.add_argument("--data", default="data")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="laion2b_s32b_b82k")
    p.add_argument("--cell", type=int, default=128)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=_index)

    p = sub.add_parser("build")
    p.add_argument("--tile", type=int, default=None)
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.3, dest="min_dist")
    p.add_argument("--reproject", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--date-from", dest="date_from", default=None, help="начало периода, ГГГГ-ММ-ДД")
    p.add_argument("--date-to", dest="date_to", default=None, help="конец периода, ГГГГ-ММ-ДД")
    p.add_argument("--name", default=None, help="имя карты (по умолчанию из дат); пишется в data/maps/<имя>")
    p.add_argument("--no-3d", dest="no_3d", action="store_true", help="не строить 3D-атлас")
    p.set_defaults(fn=_build)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-search", action="store_true")
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--vocab", default=None)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--map", default=None, help="показать карту data/maps/<имя> вместо полной")
    p.add_argument("--path-map", dest="path_map", action="append", default=None,
                   help="подмена пути оригиналов: 'C:\\photos=/photos' (можно несколько раз; или переменная SEMAMAP_PATH_MAP)")
    p.add_argument("--auth-user", default=os.environ.get("SEMAMAP_USER") or None,
                   help="Логин для доступа к карте (или переменная SEMAMAP_USER)")
    p.add_argument("--auth-pass", default=os.environ.get("SEMAMAP_PASS") or None,
                   help="Пароль для доступа к карте (или переменная SEMAMAP_PASS)")
    p.set_defaults(fn=_serve)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
