import sqlite3

# AUTOINCREMENT: иначе после удаления последней записи SQLite переиспользует id,
# и build по совпадению набора id подхватит чужие координаты UMAP.
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS images(
    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, mtime REAL, exif_date REAL,
    size INTEGER, bad INTEGER DEFAULT 0, emb BLOB);
CREATE TABLE IF NOT EXISTS thumbs(id INTEGER PRIMARY KEY, jpg BLOB);
CREATE TABLE IF NOT EXISTS thumbs512(id INTEGER PRIMARY KEY, jpg BLOB);
"""


def connect(path):
    return sqlite3.connect(str(path), timeout=120)


def meta_get(con, k):
    r = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else None


def meta_set(con, k, v):
    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (k, str(v)))


def migrate(con):
    """Донастраивает старую базу (без переиндексации): колонка exif_date и таблица thumbs512."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    if cols and "exif_date" not in cols:
        con.execute("ALTER TABLE images ADD COLUMN exif_date REAL")
    con.execute("CREATE TABLE IF NOT EXISTS thumbs512(id INTEGER PRIMARY KEY, jpg BLOB)")
    con.commit()
