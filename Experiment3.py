import os, glob, copy, warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support)

warnings.filterwarnings("ignore")

# ============================================================== CONFIG
METADATA_CSV = "combined_tracking_log.csv"
DATA_DIR     = "Dataset"
INDEX_COL    = "Index"              # here Index already holds the filename (as in your last version)
LABEL_COL    = "Diagnosis"
GROUP_COLS   = ["Dataset", "Patient_ID"]

META_NUMERIC     = ["Age", "BMI", "Child Weight (kg)", "Child Height (cm)"]
META_CATEGORICAL = ["Gender", "Sound Type", "Recording Device", "Recording Mode", "Chest Location"]

# ---- Synthetic (MelGAN) data -------------------------------------------------
# Layout:  SYNTH_DIR/<Disease>/manifest.csv (+ the wav files listed in its 'file' column)
# Folder name is matched to the Diagnosis classes (case/spaces/punctuation ignored).
# Synthetic clips are ONLY ever used for TRAINING. Val/test are always real data.
SYNTH_DIR = "synthetic"

SR, DURATION = 16000, 6.0
N_SAMPLES = int(SR * DURATION)
N_FFT, HOP = 1024, 512
N_MELS, N_MFCC, N_CHROMA = 64, 40, 12

SEED, EPOCHS, BATCH, LR = 42, 60, 32, 1e-3
PATIENCE = 10
AUG_TARGET_FRACTION = 0.5
CACHE_DIR = "cache_torch"
OUT_DIR   = "outputs"
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

# ---- Experiments to run ------------------------------------------------------
# feature sets: each entry is a list; more than one entry => multi-branch fusion model
FEATURE_SETS = {
    "wave":   ["wave"],
    "mel":    ["mel"],
    "mfcc":   ["mfcc"],
    "chroma": ["chroma"],
    "fusion": ["mel", "mfcc", "chroma"],
}
# "real"        : train on real only
# "real+synth"  : train on real + GAN data, test on real
# "synth_only"  : train on GAN data only, test on real (how useful is the GAN data alone?)
DATA_MODES = ["real", "real+synth", "synth_only"]
USE_META_OPTIONS = [False, True]     # metadata is auto-skipped for synth_only (synth has no metadata)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
print("Device:", DEVICE)


# ====================================================== 1. AUDIO LOADING
def standardize_wave(y):
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        s = (len(y) - N_SAMPLES) // 2
        y = y[s:s + N_SAMPLES]
    y = y - np.mean(y)
    p = np.max(np.abs(y))
    return (y / p if p > 0 else y).astype(np.float32)


def load_wave(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return standardize_wave(y)


def load_real():
    x_path, m_path = os.path.join(CACHE_DIR, "X_wave.npy"), os.path.join(CACHE_DIR, "meta.csv")
    if os.path.exists(x_path) and os.path.exists(m_path):
        return np.load(x_path), pd.read_csv(m_path)
    df = pd.read_csv(METADATA_CSV).dropna(subset=[LABEL_COL, INDEX_COL]).reset_index(drop=True)
    df["filepath"] = df[INDEX_COL].apply(lambda i: os.path.join(DATA_DIR, str(i)))
    X, keep = [], []
    for i, p in enumerate(df["filepath"]):
        try:
            X.append(load_wave(p)); keep.append(i)
        except Exception as e:
            print(f"Skipping {p}: {e}")
    df = df.loc[keep].reset_index(drop=True)
    X = np.stack(X)
    np.save(x_path, X); df.to_csv(m_path, index=False)
    return X, df


def load_synth(class_names):
    """Returns waves + integer labels (using the real LabelEncoder classes)."""
    x_path, y_path = os.path.join(CACHE_DIR, "S_wave.npy"), os.path.join(CACHE_DIR, "S_label.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        return np.load(x_path), np.load(y_path)
    norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    cls_lookup = {norm(c): i for i, c in enumerate(class_names)}
    X, y = [], []
    for folder in sorted(d for d in os.listdir(SYNTH_DIR) if os.path.isdir(os.path.join(SYNTH_DIR, d))):
        ci = cls_lookup.get(norm(folder))
        if ci is None:
            print(f"Synthetic folder '{folder}' matches no class in {class_names} -> skipped")
            continue
        fdir = os.path.join(SYNTH_DIR, folder)
        manifest = os.path.join(fdir, "manifest.csv")
        if os.path.exists(manifest):
            files = pd.read_csv(manifest)["file"].astype(str).tolist()
        else:
            print(f"No manifest.csv in '{folder}', using all .wav files")
            files = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(fdir, "*.wav")))]
        n_ok = 0
        for f in files:
            # 'file' may be a bare filename or a path relative to the folder / SYNTH_DIR
            cands = [os.path.join(fdir, f), os.path.join(SYNTH_DIR, f), f]
            path = next((p for p in cands if os.path.isfile(p)), None)
            if path is None:
                print(f"Missing synth file: {f}"); continue
            try:
                X.append(load_wave(path)); y.append(ci); n_ok += 1
            except Exception as e:
                print(f"Skipping synth {path}: {e}")
        print(f"Synthetic '{folder}' -> '{class_names[ci]}': {n_ok}/{len(files)} files loaded")
    if not X:
        return np.zeros((0, N_SAMPLES), np.float32), np.zeros((0,), int)
    X, y = np.stack(X), np.array(y)
    np.save(x_path, X); np.save(y_path, y)
    return X, y


# ====================================================== 2. SPLIT
def split_data(y, df):
    idx = np.arange(len(y))
    cols = [c for c in GROUP_COLS if c in df.columns]
    groups = df[cols].astype(str).agg("_".join, axis=1).values if cols else None

    def fold(indices, n):
        if groups is not None:
            sp = StratifiedGroupKFold(n, shuffle=True, random_state=SEED)
            return next(sp.split(indices, y[indices], groups[indices]))
        sp = StratifiedKFold(n, shuffle=True, random_state=SEED)
        return next(sp.split(indices, y[indices]))

    rest, test = fold(idx, 7); rest, test = idx[rest], idx[test]
    tr, va = fold(rest, 6)
    return rest[tr], rest[va], test


# ====================================================== 3. METADATA
def build_meta_transformer():
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, META_NUMERIC), ("cat", cat, META_CATEGORICAL)])


def prepare_meta(df):
    m = df[META_NUMERIC + META_CATEGORICAL].copy()
    for c in META_NUMERIC:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    for c in META_CATEGORICAL:
        m[c] = m[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return m


def empty_meta(n):
    """Synthetic clips have no metadata -> all NaN -> imputed ('missing' / median + missing-indicator)."""
    return pd.DataFrame({c: [np.nan] * n for c in META_NUMERIC + META_CATEGORICAL})


# ====================================================== 4. AUGMENTATION (real train only)
def augment_wave(y):
    ops = np.random.choice(["noise", "shift", "pitch", "stretch"], size=np.random.randint(1, 3), replace=False)
    for op in ops:
        if op == "noise":
            y = y + np.random.randn(len(y)) * np.random.uniform(0.002, 0.01)
        elif op == "shift":
            y = np.roll(y, np.random.randint(-SR // 2, SR // 2))
        elif op == "pitch":
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=np.random.uniform(-2, 2))
        elif op == "stretch":
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
            y = np.pad(y, (0, max(0, N_SAMPLES - len(y))))[:N_SAMPLES]
    p = np.max(np.abs(y))
    return (y / p if p > 0 else y).astype(np.float32)


def augment_minority(X, y):
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max() * AUG_TARGET_FRACTION)
    Xn, yn, sn = [], [], []
    for c, n in zip(classes, counts):
        if n >= target:
            continue
        pool = np.where(y == c)[0]
        for _ in range(target - n):
            s = np.random.choice(pool)
            Xn.append(augment_wave(X[s])); yn.append(c); sn.append(s)
    src = np.arange(len(y))
    if Xn:
        X = np.concatenate([X, np.stack(Xn)]); y = np.concatenate([y, np.array(yn)])
        src = np.concatenate([src, np.array(sn)])
    return X, y, src


# ====================================================== 5. FEATURE EXTRACTION
def extract(X, kind):
    """X: (N, N_SAMPLES) -> (N, 1, ...) float32. kinds: wave, mel, mfcc, chroma."""
    if kind == "wave":
        return X[:, None, :].astype(np.float32)
    out = []
    for y in X:
        if kind == "mel":
            S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
            f = librosa.power_to_db(S, ref=np.max)
        elif kind == "mfcc":
            f = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        elif kind == "chroma":
            f = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_chroma=N_CHROMA)
        else:
            raise ValueError(kind)
        out.append(f)
    return np.stack(out)[:, None].astype(np.float32)


def build_features(kinds, waves_tr, waves_va, waves_te):
    """Compute features, normalise with TRAIN statistics only (not for raw waveform)."""
    F = {}
    for k in kinds:
        a, b, c = extract(waves_tr, k), extract(waves_va, k), extract(waves_te, k)
        if k != "wave":
            mu, sd = a.mean(), a.std() + 1e-8
            a, b, c = (a - mu) / sd, (b - mu) / sd, (c - mu) / sd
        F[k] = (a, b, c)
    return F


# ====================================================== 6. DATASET + MODEL
class MultiDS(Dataset):
    def __init__(self, feats, meta, y):
        self.feats, self.meta, self.y = feats, meta, y   # feats: list of arrays

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        item = [torch.from_numpy(f[i]) for f in self.feats]
        if self.meta is not None:
            item.append(torch.from_numpy(self.meta[i]))
        item.append(torch.tensor(self.y[i], dtype=torch.long))
        return tuple(item)


def conv_block_1d(i, o, k, s=1, pool=4):
    return nn.Sequential(nn.Conv1d(i, o, k, s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(),
                         nn.MaxPool1d(pool))


class Encoder1D(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.net = nn.Sequential(conv_block_1d(1, 16, 64, 4), conv_block_1d(16, 32, 3),
                                 conv_block_1d(32, 64, 3), conv_block_1d(64, emb, 3),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.out_dim = emb

    def forward(self, x): return self.net(x)


class Encoder2D(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Dropout2d(0.25))
        self.net = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, emb),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_dim = emb

    def forward(self, x): return self.net(x)


class FusionNet(nn.Module):
    """One encoder per feature (1D for wave, 2D otherwise) -> concat (+ metadata branch) -> classifier."""
    def __init__(self, kinds, n_classes, meta_dim=0):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder1D() if k == "wave" else Encoder2D() for k in kinds])
        d = sum(e.out_dim for e in self.encoders)
        self.meta = None
        if meta_dim > 0:
            self.meta = nn.Sequential(nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.2))
            d += 32
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, n_classes))

    def forward(self, feats, meta=None):
        z = [enc(f) for enc, f in zip(self.encoders, feats)]
        if self.meta is not None:
            z.append(self.meta(meta))
        return self.head(torch.cat(z, dim=1))


def unpack(batch, n_feats, use_meta):
    feats = [b.to(DEVICE) for b in batch[:n_feats]]
    meta = batch[n_feats].to(DEVICE) if use_meta else None
    return feats, meta, batch[-1].to(DEVICE)


# ====================================================== 7. TRAIN / EVALUATE
@torch.no_grad()
def predict(model, loader, n_feats, use_meta, criterion=None):
    model.eval()
    preds, labels, tot, n = [], [], 0.0, 0
    for batch in loader:
        feats, meta, y = unpack(batch, n_feats, use_meta)
        out = model(feats, meta)
        if criterion is not None:
            tot += criterion(out, y).item() * len(y); n += len(y)
        preds.append(out.argmax(1).cpu().numpy()); labels.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels), (tot / max(n, 1))


def split_report(name, y_true, y_pred, class_names, tag, save_cm=True):
    print(f"\n----- {tag} | {name} -----")
    print(classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, zero_division=0))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    if save_cm:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues")
        plt.title(f"{tag} - {name}")
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"cm_{tag}_{name}.png"), dpi=150); plt.close()
    return {"acc": accuracy_score(y_true, y_pred), "prec": p, "rec": r, "f1": f1}


def train_and_evaluate(kinds, n_classes, class_names, tr, va, te, tag):
    """tr/va/te = (list_of_feature_arrays, meta_or_None, y)."""
    use_meta = tr[1] is not None
    meta_dim = tr[1].shape[1] if use_meta else 0
    model = FusionNet(kinds, n_classes, meta_dim).to(DEVICE)

    mk = lambda d, shuffle: DataLoader(MultiDS(*d), batch_size=BATCH, shuffle=shuffle, num_workers=0)
    dl_tr, dl_tr_eval, dl_va, dl_te = mk(tr, True), mk(tr, False), mk(va, False), mk(te, False)

    present = np.unique(tr[2])
    w = np.ones(n_classes, dtype=np.float32)
    w[present] = compute_class_weight("balanced", classes=present, y=tr[2])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    n_feats = len(kinds)
    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    best_loss, best_state, bad = float("inf"), None, 0

    for ep in range(1, EPOCHS + 1):
        model.train(); tl, tc, tn = 0.0, 0, 0
        for batch in dl_tr:
            feats, meta, y = unpack(batch, n_feats, use_meta)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                out = model(feats, meta); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); tn += len(y)
        pv, lv, vloss = predict(model, dl_va, n_feats, use_meta, criterion)
        vacc = accuracy_score(lv, pv)
        sched.step(vloss)
        hist["tr_loss"].append(tl / tn); hist["tr_acc"].append(tc / tn)
        hist["va_loss"].append(vloss); hist["va_acc"].append(vacc)
        print(f"[{tag}] ep {ep:02d} | train loss {tl/tn:.4f} acc {tc/tn:.4f} | val loss {vloss:.4f} acc {vacc:.4f}")
        if vloss < best_loss - 1e-4:
            best_loss, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping."); break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, f"model_{tag}.pt"))

    # learning curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["va_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["tr_acc"], label="train"); ax[1].plot(hist["va_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"curves_{tag}.png"), dpi=150); plt.close()

    # ---- train / validation / test evaluation AFTER training (best checkpoint) ----
    res = {"run": tag}
    for name, loader, d in (("train", dl_tr_eval, tr), ("val", dl_va, va), ("test", dl_te, te)):
        p, l, _ = predict(model, loader, n_feats, use_meta)
        m = split_report(name, l, p, class_names, tag)
        for k, v in m.items():
            res[f"{name}_{k}"] = v
    return res


# ====================================================== MAIN
if __name__ == "__main__":
    X, df = load_real()
    le = LabelEncoder(); y = le.fit_transform(df[LABEL_COL].astype(str).str.strip())
    class_names = list(le.classes_); n_classes = len(class_names)
    print("Class counts:", dict(zip(class_names, np.bincount(y))))

    tr, va, te = split_data(y, df)
    print(f"Split -> train {len(tr)}, val {len(va)}, test {len(te)}")

    # metadata (fit on real train only)
    meta_df = prepare_meta(df)
    mt = build_meta_transformer().fit(meta_df.iloc[tr])
    M_tr0, M_va, M_te = (mt.transform(meta_df.iloc[i]).astype(np.float32) for i in (tr, va, te))

    # real train: minority augmentation (metadata copied from source clip)
    Xtr_real, ytr_real, src = augment_minority(X[tr], y[tr])
    Mtr_real = M_tr0[src]
    Xva, yva, Xte, yte = X[va], y[va], X[te], y[te]

    # synthetic (train-only, no metadata)
    Xs, ys = load_synth(class_names)
    Ms = mt.transform(prepare_meta(empty_meta(len(Xs)))).astype(np.float32) if len(Xs) else np.zeros((0, M_tr0.shape[1]), np.float32)
    print(f"Real train (after aug): {len(ytr_real)} | Synthetic: {len(ys)}")
    if len(ys):
        print("Synthetic class counts:", dict(zip(class_names, np.bincount(ys, minlength=n_classes))))

    results = []
    for mode in DATA_MODES:
        if mode != "real" and len(Xs) == 0:
            print(f"Skipping '{mode}': no synthetic data found in {SYNTH_DIR}"); continue
        if mode == "real":
            Xt, yt, Mt = Xtr_real, ytr_real, Mtr_real
        elif mode == "real+synth":
            Xt = np.concatenate([Xtr_real, Xs]); yt = np.concatenate([ytr_real, ys])
            Mt = np.concatenate([Mtr_real, Ms])
        else:
            Xt, yt, Mt = Xs, ys, Ms

        for fs_name, kinds in FEATURE_SETS.items():
            F = build_features(kinds, Xt, Xva, Xte)       # features computed once per (mode, feature set)
            for use_meta in USE_META_OPTIONS:
                if use_meta and mode == "synth_only":
                    continue
                tag = f"{mode.replace('+', '_')}__{fs_name}__{'audio_meta' if use_meta else 'audio_only'}"
                print(f"\n################ {tag} ################")
                D = lambda i, M, yy: ([F[k][i] for k in kinds], M if use_meta else None, yy)
                r = train_and_evaluate(kinds, n_classes, class_names,
                                       D(0, Mt, yt), D(1, M_va, yva), D(2, M_te, yte), tag)
                r.update({"data_mode": mode, "features": fs_name, "meta": use_meta})
                results.append(r)
                pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            del F

    comp = pd.DataFrame(results)
    cols = ["data_mode", "features", "meta", "train_acc", "val_acc", "test_acc", "test_prec", "test_rec", "test_f1"]
    print("\n", comp[cols].round(4).to_string(index=False))