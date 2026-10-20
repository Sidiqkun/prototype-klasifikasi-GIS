"""
Perbandingan metode klasifikasi citra drone untuk pemetaan lahan pertanian.

Metode : Random Forest (RF), SVM-RBF, CNN patch sederhana
Input  : orthomosaic (GeoTIFF/PNG/JPG) + raster label (0 = tidak berlabel, 1..K = kelas)
Output : folder hasil/ berisi results.csv, report.txt, confusion_matrix.png, peta_hasil.png

Contoh:
  python bandingkan.py --demo
  python bandingkan.py --image ortho.tif --labels label.tif --names "Padi,Jagung,Tanah,Bangunan" --scale 0.25 --map
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import binary_dilation, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ----------------------------------------------------------------- data
def load_raster(path, out_hw=None, scale=1.0, nearest=False):
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as src:
        h, w = out_hw if out_hw else (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
        rs = Resampling.nearest if nearest else Resampling.bilinear
        return src.read(out_shape=(src.count, h, w), resampling=rs)


def stretch(img, valid):
    """Regangan kontras per band (persentil 1-99) ke 0..1."""
    out = np.zeros(img.shape, np.float32)
    for i in range(img.shape[0]):
        lo, hi = np.percentile(img[i][valid], (1, 99))
        out[i] = np.clip((img[i] - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def make_demo(size=500, seed=0):
    """Data sintetis 4 kelas (padi, jagung, tanah, bangunan) untuk uji pipeline."""
    rng = np.random.default_rng(seed)
    pts = rng.random((14, 2)) * size
    seed_cls = rng.integers(1, 5, 14)
    seed_cls[:4] = [1, 2, 3, 4]
    yy, xx = np.mgrid[:size, :size]
    d = (yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2
    nearest = d.argmin(-1)
    lab = seed_cls[nearest]
    base = {1: (60, 140, 50), 2: (85, 150, 60), 3: (150, 120, 90), 4: (170, 170, 175)}
    img = np.zeros((3, size, size), np.float32)
    shift = rng.normal(0, 8, 14)[nearest]
    stripes = 1 + 0.12 * np.sin(xx * 0.8) * (lab == 2)
    for c in range(3):
        v = np.choose(lab - 1, [base[k][c] for k in (1, 2, 3, 4)])
        img[c] = (v + shift) * stripes + rng.normal(0, 12, (size, size))
    # sebagian area tidak berlabel
    blocks = rng.random((size // 25 + 1, size // 25 + 1)) < 0.1
    lab[blocks[yy // 25, xx // 25]] = 0
    return np.clip(img, 0, 255), lab


# ------------------------------------------------------------- fitur RF/SVM
def build_features(img, nir):
    r, g, b = img[0], img[1], img[2]
    eps = 1e-6
    s = r + g + b + eps
    exg = 2 * g / s - r / s - b / s
    vari = np.clip(np.nan_to_num((g - r) / (g + r - b + eps)), -1, 1)
    feats = [*img, exg, vari]
    if nir:
        feats.append((img[3] - r) / (img[3] + r + eps))
    gray = img[:3].mean(0)
    for k in (5, 11):  # tekstur: simpangan baku lokal
        m, m2 = uniform_filter(gray, k), uniform_filter(gray ** 2, k)
        feats.append(np.sqrt(np.maximum(m2 - m * m, 0)))
    return np.stack(feats, -1).astype(np.float32)


# --------------------------------------------------------- split & sampling
def split_blocks(y, K, block, test_frac, min_n, seed, buffer):
    """Split spasial per blok agar data uji tidak 'bocor' ke data latih."""
    h, w = y.shape
    nby, nbx = -(-h // block), -(-w // block)
    ri, ci = (np.arange(h) // block)[:, None], (np.arange(w) // block)[None, :]
    for s in range(seed, seed + 200):
        tb = np.random.default_rng(s).random((nby, nbx)) < test_frac
        is_test = tb[ri, ci]
        train_ok = ~binary_dilation(is_test, iterations=buffer)  # zona penyangga
        if all(((y == k) & is_test).sum() >= min_n and ((y == k) & train_ok).sum() >= min_n
               for k in range(K)):
            return is_test, train_ok
    raise SystemExit("Tidak bisa membagi data: coba --block lebih kecil atau --test-frac lain, "
                     "atau pastikan tiap kelas cukup luas.")


def sample(y, region, K, n_per_class, rng):
    rr, cc = [], []
    for k in range(K):
        r, c = np.nonzero((y == k) & region)
        t = rng.choice(len(r), min(n_per_class, len(r)), replace=False)
        rr.append(r[t]); cc.append(c[t])
    rr, cc = np.concatenate(rr), np.concatenate(cc)
    return rr, cc, y[rr, cc]


# ------------------------------------------------------------------ metode
def run_sklearn(name, model, F, tr, te, valid, scale_feat, want_map):
    Xtr, Xte = F[tr[0], tr[1]], F[te[0], te[1]]
    sc = StandardScaler().fit(Xtr) if scale_feat else None
    f = (lambda X: sc.transform(X)) if sc else (lambda X: X)
    t = time.time(); model.fit(f(Xtr), tr[2]); train_s = time.time() - t
    t = time.time(); pred = model.predict(f(Xte)); infer_s = time.time() - t
    pmap, map_s = None, np.nan
    if want_map:
        t = time.time()
        flat = F.reshape(-1, F.shape[-1])
        out = np.concatenate([model.predict(f(flat[i:i + 200_000]))
                              for i in range(0, len(flat), 200_000)])
        pmap, map_s = out.reshape(F.shape[:2]), time.time() - t
    return pred, train_s, infer_s, pmap, map_s


def run_cnn(img, valid, tr, te, K, args, want_map):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  CNN memakai: {dev}")
    torch.manual_seed(args.seed); random.seed(args.seed)
    p, C = args.patch, img.shape[0]
    mean = img[:, valid].mean(1)[:, None, None]
    std = img[:, valid].std(1)[:, None, None] + 1e-6
    x = ((img - mean) / std).transpose(1, 2, 0)  # H,W,C
    pad = p // 2
    win = sliding_window_view(np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"),
                              (p, p), axis=(0, 1))  # H,W,C,p,p (view, hemat memori)

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    model = nn.Sequential(block(C, 16), nn.MaxPool2d(2), block(16, 32), nn.MaxPool2d(2),
                          block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                          nn.Dropout(0.3), nn.Linear(64, K)).to(dev)

    Xtr = torch.from_numpy(np.ascontiguousarray(win[tr[0], tr[1]]))
    ytr = torch.from_numpy(tr[2]).long()
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    t = time.time()
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(ytr)); tot = 0
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb, yb = Xtr[b].to(dev), ytr[b].to(dev)
            if random.random() < 0.5: xb = torch.flip(xb, [3])
            if random.random() < 0.5: xb = torch.flip(xb, [2])
            xb = torch.rot90(xb, random.randint(0, 3), [2, 3])
            loss = Fn.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep + 1}/{args.epochs}  loss={tot / len(ytr):.4f}")
    train_s = time.time() - t

    @torch.no_grad()
    def predict(patches):
        model.eval(); out = []
        for i in range(0, len(patches), 2048):
            xb = torch.from_numpy(np.ascontiguousarray(patches[i:i + 2048])).to(dev)
            out.append(model(xb).argmax(1).cpu().numpy())
        return np.concatenate(out)

    t = time.time(); pred = predict(win[te[0], te[1]]); infer_s = time.time() - t
    pmap, map_s = None, np.nan
    if want_map:
        t = time.time(); H, W = valid.shape
        pmap = np.concatenate([predict(win[r:r + 16].reshape(-1, C, p, p)).reshape(-1, W)
                               for r in range(0, H, 16)])
        map_s = time.time() - t
    return pred, train_s, infer_s, pmap, map_s


# -------------------------------------------------------------------- plot
def colorize(m, K):
    cmap = plt.get_cmap("tab10")
    rgb = np.full((*m.shape, 3), 0.85)
    for k in range(K):
        rgb[m == k] = cmap(k)[:3]
    return rgb


def save_plots(out, img, y, valid, names, results, maps, cms):
    K = len(names)
    n = len(cms)
    fig, axs = plt.subplots(1, n, figsize=(4.5 * n, 4.2), squeeze=False)
    for ax, (m, cm) in zip(axs[0], cms.items()):
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > .5 else "black", fontsize=8)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
        ax.set_title(f"{m}\nOA={results[m]['OA']:.3f}"); ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    fig.tight_layout(); fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150); plt.close(fig)

    if maps:
        panels = [("Citra (RGB)", img[:3].transpose(1, 2, 0)), ("Label referensi", colorize(y, K))]
        panels += [(m, colorize(np.where(valid, p, -1), K)) for m, p in maps.items()]
        fig, axs = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8), squeeze=False)
        for ax, (t, a) in zip(axs[0], panels):
            ax.imshow(a); ax.set_title(t); ax.axis("off")
        fig.legend(handles=[Patch(color=plt.get_cmap("tab10")(k), label=n_) for k, n_ in enumerate(names)],
                   loc="lower center", ncol=K)
        fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(out, "peta_hasil.png"), dpi=150)
        plt.close(fig)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image"); ap.add_argument("--labels")
    ap.add_argument("--demo", action="store_true", help="pakai data sintetis untuk uji")
    ap.add_argument("--names", help="nama kelas urut sesuai nilai label, pisah koma")
    ap.add_argument("--nir", action="store_true", help="band ke-4 adalah NIR (bukan alpha)")
    ap.add_argument("--scale", type=float, default=1.0, help="skala resolusi, mis. 0.25")
    ap.add_argument("--methods", default="rf,svm,cnn")
    ap.add_argument("--block", type=int, default=50, help="ukuran blok split spasial (piksel)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--n-train", type=int, default=3000, help="sampel latih per kelas")
    ap.add_argument("--n-test", type=int, default=3000, help="sampel uji per kelas")
    ap.add_argument("--patch", type=int, default=15, help="ukuran patch CNN (ganjil)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--map", action="store_true", help="buat peta klasifikasi penuh")
    ap.add_argument("--out", default="hasil")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(",")]

    # --- muat data
    if args.demo:
        raw, lab = make_demo(seed=args.seed)
        if not args.names: args.names = "Padi,Jagung,Tanah,Bangunan"
        nir = False
    else:
        if not (args.image and args.labels):
            ap.error("butuh --image dan --labels (atau pakai --demo)")
        raw = load_raster(args.image, scale=args.scale).astype(np.float32)
        nir = args.nir
        raw = raw[:4] if nir else raw[:3]
        if raw.shape[0] < (4 if nir else 3):
            ap.error("citra harus punya minimal 3 band (RGB)")
        lab = load_raster(args.labels, out_hw=raw.shape[1:], nearest=True)[0].astype(int)
    valid = ~(raw == 0).all(0)
    if valid.sum() == 0: raise SystemExit("Semua piksel kosong/0.")
    img = stretch(raw, valid)

    classes = np.unique(lab[(lab > 0) & valid])
    K = len(classes)
    if K < 2: raise SystemExit("Label harus berisi minimal 2 kelas (nilai > 0).")
    y = np.full(lab.shape, -1, int)
    for i, c in enumerate(classes): y[lab == c] = i
    y[~valid] = -1
    names = args.names.split(",") if args.names else [f"kelas_{c}" for c in classes]
    if len(names) != K: raise SystemExit(f"--names berisi {len(names)} nama, tetapi ada {K} kelas: {list(classes)}")
    print(f"Citra {img.shape[2]}x{img.shape[1]} px, {K} kelas: {names}")

    # --- split spasial & sampling
    rng = np.random.default_rng(args.seed)
    is_test, train_ok = split_blocks(y, K, args.block, args.test_frac, 50, args.seed, args.patch // 2 + 1)
    tr = sample(y, train_ok, K, args.n_train, rng)
    te = sample(y, is_test, K, args.n_test, rng)
    print(f"Sampel latih={len(tr[2])}, uji={len(te[2])} (dipisah per blok spasial)")

    # --- jalankan metode
    t = time.time(); F = build_features(img, nir); feat_s = time.time() - t
    n_valid = int(valid.sum())
    results, maps, cms, report = {}, {}, {}, []
    for m in methods:
        print(f"[{m.upper()}]")
        if m == "rf":
            out = run_sklearn("RF", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=args.seed),
                              F, tr, te, valid, False, args.map); label, fs = "Random Forest", feat_s
        elif m == "svm":
            out = run_sklearn("SVM", SVC(kernel="rbf", C=10, gamma="scale"),
                              F, tr, te, valid, True, args.map); label, fs = "SVM (RBF)", feat_s
        elif m == "cnn":
            out = run_cnn(img, valid, tr, te, K, args, args.map); label, fs = "CNN patch", 0.0
        else:
            raise SystemExit(f"Metode tidak dikenal: {m}")
        pred, train_s, infer_s, pmap, map_s = out
        yt = te[2]
        results[label] = dict(
            OA=float((pred == yt).mean()), Kappa=cohen_kappa_score(yt, pred),
            F1_macro=f1_score(yt, pred, average="macro"),
            fitur_s=fs, latih_s=train_s, infer_uji_s=infer_s,
            est_peta_s=fs + infer_s / len(yt) * n_valid, peta_aktual_s=map_s)
        cms[label] = confusion_matrix(yt, pred, labels=range(K))
        report.append(f"=== {label} ===\n" + classification_report(
            yt, pred, labels=range(K), target_names=names, zero_division=0, digits=3))
        if pmap is not None: maps[label] = pmap
        r = results[label]
        print(f"  OA={r['OA']:.3f} Kappa={r['Kappa']:.3f} F1={r['F1_macro']:.3f} latih={train_s:.1f}s")

    # --- simpan & rekomendasi
    df = pd.DataFrame(results).T.round(4)
    df.insert(0, "n_latih", len(tr[2])); df.insert(1, "n_uji", len(te[2]))
    df.index.name = "metode"
    df.to_csv(os.path.join(args.out, "results.csv"))
    with open(os.path.join(args.out, "report.txt"), "w") as f: f.write("\n".join(report))
    save_plots(args.out, img, y, valid, names, results, maps, cms)

    best = df["F1_macro"].max()
    cand = df[df["F1_macro"] >= best - 0.01]  # selisih F1 <= 1 poin dianggap setara
    eff = (cand["latih_s"] + cand["est_peta_s"]).idxmin()
    print("\n" + df.drop(columns=["n_latih", "n_uji"]).to_string())
    print(f"\nMetode paling efisien (akurasi setara terbaik, total waktu latih+peta terkecil): {eff}")
    print(f"Hasil tersimpan di '{args.out}/'")


if __name__ == "__main__":
    main()
