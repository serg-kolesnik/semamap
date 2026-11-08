"""Теги, NSFW-сигнал и подписи районов карты."""
import math

import numpy as np

from .config import PROMPT_TEMPLATES


def encode_vocab(model, tok, vocab):
    import torch
    acc = None
    with torch.no_grad():
        for tmpl in PROMPT_TEMPLATES:
            prompts = [tmpl.format(en) for en, _ in vocab]
            chunks = []
            for i in range(0, len(prompts), 64):
                t = model.encode_text(tok(prompts[i:i + 64]))
                chunks.append(torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy())
            arr = np.concatenate(chunks)
            acc = arr if acc is None else acc + arr
    acc /= len(PROMPT_TEMPLATES)
    acc /= np.linalg.norm(acc, axis=1, keepdims=True) + 1e-8
    return acc.astype(np.float32)


def softmax_scores(V, v, scale=25.0):
    lg = scale * (V @ v)
    lg = lg - lg.max()
    p = np.exp(lg)
    return p / p.sum()


def _top_tags(E, V, allowed, top_n, min_cos, chunk=4096):
    """Топ-N тегов из `allowed` для каждой картинки; считаем кусками, чтобы не раздувать память."""
    if len(allowed) == 0:
        return [[] for _ in range(len(E))]
    allowed = np.asarray(allowed)
    Vs = V[allowed]
    k = min(top_n, len(allowed))
    out = []
    for s in range(0, len(E), chunk):
        sc = E[s:s + chunk] @ Vs.T
        idx = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        for r in range(len(sc)):
            row = idx[r][np.argsort(-sc[r, idx[r]])]
            out.append([int(allowed[j]) for j in row if sc[r, j] >= min_cos])
    return out


def precompute_image_tags(E, V, nsfw_mask, top_n=12, min_cos=0.15):
    """SFW-теги по порогу cosine."""
    return _top_tags(E, V, np.nonzero(~nsfw_mask)[0], top_n, min_cos)


def precompute_nsfw_tags(E, V, nsfw_idx, top_n=5, min_cos=0.18):
    return _top_tags(E, V, nsfw_idx, top_n, min_cos)


def region_labels(grid, E, V, names, bs, nsfw_mask=None, nsfw_signal=None, min_p=0.18):
    H, W = grid.shape
    by, bx = math.ceil(H / bs), math.ceil(W / bs)
    lab = np.full((by, bx), -1, dtype=np.int64)
    cnt = np.zeros((by, bx), dtype=np.int64)
    for j in range(by):
        for i in range(bx):
            sub = grid[j * bs:(j + 1) * bs, i * bs:(i + 1) * bs].ravel()
            sub = sub[sub >= 0]
            if len(sub) < max(2, bs * bs // 6):
                continue
            m = E[sub].mean(0)
            nrm = float(np.linalg.norm(m))
            if nrm == 0:
                continue
            region_nsfw = False
            if nsfw_signal is not None:
                region_nsfw = float(nsfw_signal[sub].mean()) > 0.015
            pr = softmax_scores(V, m / nrm)
            for k in np.argsort(-pr):
                if pr[k] < min_p:
                    break
                if nsfw_mask is not None and nsfw_mask[k] and not region_nsfw:
                    continue
                lab[j, i] = k
                cnt[j, i] = len(sub)
                break
    items, seen = [], np.zeros(lab.shape, dtype=bool)
    for j0 in range(by):
        for i0 in range(bx):
            if lab[j0, i0] < 0 or seen[j0, i0]:
                continue
            k, comp, stack = int(lab[j0, i0]), [], [(j0, i0)]
            seen[j0, i0] = True
            while stack:
                j, i = stack.pop()
                comp.append((j, i))
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < by and 0 <= ii < bx and not seen[jj, ii] and lab[jj, ii] == k:
                        seen[jj, ii] = True
                        stack.append((jj, ii))
            w = np.array([cnt[j, i] for j, i in comp], dtype=np.float64)
            cj = float(np.dot(w, [j for j, _ in comp]) / w.sum())
            ci = float(np.dot(w, [i for _, i in comp]) / w.sum())
            j, i = min(comp, key=lambda t: (t[0] - cj) ** 2 + (t[1] - ci) ** 2)
            items.append({"x": round((i * bs + min((i + 1) * bs, W)) / 2, 1),
                          "y": round((j * bs + min((j + 1) * bs, H)) / 2, 1),
                          "t": names[k], "n": len(comp)})
    return items


def build_labels(S):
    m = max(S.W, S.H)
    names = [ru for _, ru in S.vocab]
    return [{"bs": bs, "items": region_labels(
        S.grid, S.E, S.V, names, bs,
        getattr(S, "nsfw_mask", None), getattr(S, "nsfw_signal", None))}
        for bs in sorted({max(3, m // 24), max(6, m // 8)})]
