import os, glob, copy, warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score,
                             balanced_accuracy_score, precision_recall_fscore_support)

warnings.filterwarnings("ignore")

# ============================================================== CONFIG
METADATA_CSV = "combined_tracking_log.csv"
DATA_DIR     = "Dataset"
INDEX_COL    = "Index"              # holds the wav filename
LABEL_COL    = "Diagnosis"

# Merge small classes. Keys = original Diagnosis values, values = merged class name.
# Anything not listed keeps its own name. Set CLASS_MAP = {} to go back to the original 6 classes.
CLASS_MAP = {"Asthma": "Other", "Bronchiolitis": "Other", "Pneumonia": "Other", "URTI": "Other"}
_norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
_MAP_NORM = {_norm(k): v for k, v in CLASS_MAP.items()}
merge_label = lambda s: _MAP_NORM.get(_norm(s), str(s).strip())
GROUP_COLS   = ["Dataset", "Patient_ID"]

META_NUMERIC     = ["Age", "BMI", "Child Weight (kg)", "Child Height (cm)"]
META_CATEGORICAL = ["Gender", "Sound Type", "Recording Device", "Recording Mode", "Chest Location"]

# Synthetic (MelGAN): SYNTH_DIR/<Disease>/manifest.csv (+ wavs listed in 'file' column)
SYNTH_DIR = "synthetic"
# Leakage filter: synthetic clips whose source_a/source_b is a real clip of a VAL/TEST patient are dropped.
# Sources that cannot be matched to a real clip are dropped too unless this is True.
SYNTH_KEEP_UNMATCHED = False

SR, DURATION = 16000, 6.0
N_SAMPLES = int(SR * DURATION)
N_FFT, HOP = 1024, 512
N_MELS, N_MFCC, N_CHROMA = 64, 40, 12

SPLIT_SEED = 42                     # data split is FIXED; only training randomness changes between seeds
SEEDS      = [42, 43, 44]           # repeated runs -> mean +- std
EPOCHS, PATIENCE, BATCH, LR = 30, 6, 32, 1e-3
AUG_TARGET_FRACTION = 0.25          # lowered: the sampler below already balances classes (and this is faster)
BALANCE = "sampler"                 # "sampler" | "weights" | "focal"   (one mechanism only, not stacked)

CACHE_DIR, OUT_DIR = "cache_torch", "outputs_v2_merged"
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

# Reduced experiment grid (24 -> 8 configs x len(SEEDS) runs)
FEATURE_SETS = {"mel": ["mel"], "fusion": ["mel", "mfcc", "chroma"]}
# other options: "wave": ["wave"], "mfcc": ["mfcc"], "chroma": ["chroma"]
DATA_MODES = ["real", "real+synth"]          # "synth_only" also possible
USE_META_OPTIONS = [False, True]

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", DEVICE)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(SPLIT_SEED)


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
    """Returns waves, labels, and an info DataFrame with the source_a / source_b of every clip."""
    suffix = "_".join(_norm(c) for c in class_names)     # cache depends on the class set
    x_path = os.path.join(CACHE_DIR, f"S2_wave_{suffix}.npy")
    i_path = os.path.join(CACHE_DIR, f"S2_info_{suffix}.csv")
    if os.path.exists(x_path) and os.path.exists(i_path):
        info = pd.read_csv(i_path)
        return np.load(x_path), info["label"].values.astype(int), info
    cls_lookup = {_norm(c): i for i, c in enumerate(class_names)}
    X, rows = [], []
    for folder in sorted(d for d in os.listdir(SYNTH_DIR) if os.path.isdir(os.path.join(SYNTH_DIR, d))):
        ci = cls_lookup.get(_norm(merge_label(folder)))   # e.g. folder 'asthma' -> 'Other'
        if ci is None:
            print(f"Synthetic folder '{folder}' matches no class in {class_names} -> skipped"); continue
        fdir = os.path.join(SYNTH_DIR, folder)
        manifest = os.path.join(fdir, "manifest.csv")
        if os.path.exists(manifest):
            man = pd.read_csv(manifest)
        else:
            print(f"No manifest.csv in '{folder}', using all .wav files (no source info -> leakage filter can't run)")
            man = pd.DataFrame({"file": [os.path.basename(p) for p in sorted(glob.glob(os.path.join(fdir, "*.wav")))]})
        n_ok = 0
        for _, r in man.iterrows():
            f = str(r["file"])
            path = next((p for p in (os.path.join(fdir, f), os.path.join(SYNTH_DIR, f), f) if os.path.isfile(p)), None)
            if path is None:
                print(f"Missing synth file: {f}"); continue
            try:
                X.append(load_wave(path)); n_ok += 1
                rows.append({"label": ci, "file": f, "source_a": r.get("source_a", np.nan),
                             "source_b": r.get("source_b", np.nan)})
            except Exception as e:
                print(f"Skipping synth {path}: {e}")
        print(f"Synthetic '{folder}' -> '{class_names[ci]}': {n_ok}/{len(man)} files loaded")
    if not X:
        return (np.zeros((0, N_SAMPLES), np.float32), np.zeros((0,), int),
                pd.DataFrame(columns=["label", "file", "source_a", "source_b"]))
    X, info = np.stack(X), pd.DataFrame(rows)
    np.save(x_path, X); info.to_csv(i_path, index=False)
    return X, info["label"].values.astype(int), info


# ====================================================== 2. SPLIT + LEAKAGE FILTER
def get_groups(df):
    cols = [c for c in GROUP_COLS if c in df.columns]
    return df[cols].astype(str).agg("_".join, axis=1).values if cols else None


def split_data(y, df):
    idx = np.arange(len(y)); groups = get_groups(df)

    def fold(indices, n):
        if groups is not None:
            sp = StratifiedGroupKFold(n, shuffle=True, random_state=SPLIT_SEED)
            return next(sp.split(indices, y[indices], groups[indices]))
        sp = StratifiedKFold(n, shuffle=True, random_state=SPLIT_SEED)
        return next(sp.split(indices, y[indices]))

    rest, test = fold(idx, 7); rest, test = idx[rest], idx[test]
    tr, va = fold(rest, 6)
    return rest[tr], rest[va], test


def stem_key(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    return os.path.splitext(os.path.basename(str(s).replace("\\", "/")))[0].strip().lower()


def filter_synth(Xs, ys, info, df, va, te):
    """Drop synthetic clips built from real clips of val/test patients."""
    if len(ys) == 0:
        return Xs, ys, info
    groups = get_groups(df)
    row_of = {stem_key(v): i for i, v in enumerate(df[INDEX_COL])}
    held_rows = set(int(i) for i in np.concatenate([va, te]))
    held_groups = set(groups[sorted(held_rows)]) if groups is not None else set()

    keep = np.ones(len(ys), bool)
    n_leak = n_unmatched = n_sources = n_matched = 0
    for k, (a, b) in enumerate(zip(info["source_a"], info["source_b"])):
        leak = unmatched = False
        for s in (a, b):
            key = stem_key(s)
            if key is None:
                continue
            n_sources += 1
            r = row_of.get(key)
            if r is None:
                unmatched = True; continue
            n_matched += 1
            if r in held_rows or (groups is not None and groups[r] in held_groups):
                leak = True
        if leak:
            keep[k] = False; n_leak += 1
        elif unmatched:
            n_unmatched += 1
            if not SYNTH_KEEP_UNMATCHED:
                keep[k] = False
    print(f"\n[Leakage filter] synthetic clips: {len(ys)} | dropped (val/test source): {n_leak} | "
          f"unmatched sources: {n_unmatched} ({'kept' if SYNTH_KEEP_UNMATCHED else 'dropped'}) | kept: {int(keep.sum())}")
    if n_sources == 0 or n_matched == 0:
        print("  !! No source_a/source_b could be matched to a real clip. Check their format against the "
              f"'{INDEX_COL}' column.\n     example source: {info['source_a'].iloc[0]!r} | example Index: {df[INDEX_COL].iloc[0]!r}")
    return Xs[keep], ys[keep], info[keep].reset_index(drop=True)


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
    out = {}
    for k in kinds:
        a, b, c = extract(waves_tr, k), extract(waves_va, k), extract(waves_te, k)
        if k != "wave":
            mu, sd = a.mean(), a.std() + 1e-8
            a, b, c = (a - mu) / sd, (b - mu) / sd, (c - mu) / sd
        out[k] = (a, b, c)
    return out


# ====================================================== 6. DATASET + MODEL
class MultiDS(Dataset):
    def __init__(self, feats, meta, y):
        self.feats, self.meta, self.y = feats, meta, y

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        item = [torch.from_numpy(f[i]) for f in self.feats]
        if self.meta is not None:
            item.append(torch.from_numpy(self.meta[i]))
        item.append(torch.tensor(self.y[i], dtype=torch.long))
        return tuple(item)


def conv_block_1d(i, o, k, s=1, pool=4):
    return nn.Sequential(nn.Conv1d(i, o, k, s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(pool))


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
        self.net = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, emb), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_dim = emb

    def forward(self, x): return self.net(x)


class FusionNet(nn.Module):
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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


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


def metrics_of(y_true, y_pred, n_classes):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": accuracy_score(y_true, y_pred), "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "prec": p, "rec": r, "f1": f1}


def split_report(name, y_true, y_pred, class_names, tag, verbose=True):
    n = len(class_names)
    m = metrics_of(y_true, y_pred, n)
    if verbose:
        print(f"\n----- {tag} | {name} -----")
        print(classification_report(y_true, y_pred, labels=range(n), target_names=class_names, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=range(n))
        plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues"); plt.title(f"{tag} - {name}")
        plt.xticks(range(n), class_names, rotation=45); plt.yticks(range(n), class_names)
        for i in range(n):
            for j in range(n):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"cm_{tag}_{name}.png"), dpi=130); plt.close()
    return m


def train_and_evaluate(kinds, n_classes, class_names, tr, va, te, tag, seed):
    set_seed(seed)
    use_meta = tr[1] is not None
    model = FusionNet(kinds, n_classes, tr[1].shape[1] if use_meta else 0).to(DEVICE)

    ds_tr = MultiDS(*tr)
    if BALANCE == "sampler":
        cnt = np.bincount(tr[2], minlength=n_classes).astype(float)
        sw = 1.0 / np.maximum(cnt[tr[2]], 1.0)
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), num_samples=len(tr[2]), replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, sampler=sampler, num_workers=0)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0)
    dl_tr_eval = DataLoader(MultiDS(*tr), batch_size=BATCH, shuffle=False)
    dl_va = DataLoader(MultiDS(*va), batch_size=BATCH, shuffle=False)
    dl_te = DataLoader(MultiDS(*te), batch_size=BATCH, shuffle=False)

    if BALANCE == "weights":
        from sklearn.utils.class_weight import compute_class_weight
        present = np.unique(tr[2]); w = np.ones(n_classes, dtype=np.float32)
        w[present] = compute_class_weight("balanced", classes=present, y=tr[2])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE))
    elif BALANCE == "focal":
        criterion = FocalLoss(2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    n_feats = len(kinds)
    hist = {"tr_loss": [], "va_loss": [], "va_f1": []}
    best_f1, best_state, bad = -1.0, None, 0

    for ep in range(1, EPOCHS + 1):
        model.train(); tl, tn = 0.0, 0
        for batch in dl_tr:
            feats, meta, y = unpack(batch, n_feats, use_meta)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                out = model(feats, meta); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); tn += len(y)
        pv, lv, vloss = predict(model, dl_va, n_feats, use_meta, criterion)
        vf1 = f1_score(lv, pv, average="macro", zero_division=0)
        sched.step(vf1)
        hist["tr_loss"].append(tl / tn); hist["va_loss"].append(vloss); hist["va_f1"].append(vf1)
        print(f"[{tag}] ep {ep:02d} | train loss {tl/tn:.4f} | val loss {vloss:.4f} | val acc {accuracy_score(lv, pv):.4f} | val macroF1 {vf1:.4f}")
        if vf1 > best_f1 + 1e-4:
            best_f1, best_state, bad = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping (val macro-F1)."); break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, f"model_{tag}.pt"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["va_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["va_f1"]); ax[1].set_title("Val macro-F1")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"curves_{tag}.png"), dpi=130); plt.close()

    res = {"run": tag, "seed": seed}
    for name, loader, verbose in (("train", dl_tr_eval, False), ("val", dl_va, True), ("test", dl_te, True)):
        p, l, _ = predict(model, loader, n_feats, use_meta)
        m = split_report(name, l, p, class_names, tag, verbose)
        for k, v in m.items():
            res[f"{name}_{k}"] = v
        if name == "test":   # per-class recall (sensitivity) for the write-up
            _, rec, _, sup = precision_recall_fscore_support(l, p, labels=range(n_classes), zero_division=0)
            for ci, cname in enumerate(class_names):
                res[f"test_recall_{cname}"] = rec[ci] if sup[ci] > 0 else np.nan
    return res


# ====================================================== MAIN
if __name__ == "__main__":
    X, df = load_real()
    merged = df[LABEL_COL].astype(str).str.strip().map(merge_label)
    le = LabelEncoder(); y = le.fit_transform(merged)
    class_names = list(le.classes_); n_classes = len(class_names)
    print("Class counts:", dict(zip(class_names, np.bincount(y))))

    tr, va, te = split_data(y, df)
    print(f"Split -> train {len(tr)}, val {len(va)}, test {len(te)}")
    print("Test class counts:", dict(zip(class_names, np.bincount(y[te], minlength=n_classes))))

    meta_df = prepare_meta(df)
    mt = build_meta_transformer().fit(meta_df.iloc[tr])
    M_tr0, M_va, M_te = (mt.transform(meta_df.iloc[i]).astype(np.float32) for i in (tr, va, te))

    Xtr_real, ytr_real, src = augment_minority(X[tr], y[tr])
    Mtr_real = M_tr0[src]
    Xva, yva, Xte, yte = X[va], y[va], X[te], y[te]

    # synthetic: load -> leakage filter -> metadata placeholders
    if os.path.isdir(SYNTH_DIR):
        Xs, ys, sinfo = load_synth(class_names)
        Xs, ys, sinfo = filter_synth(Xs, ys, sinfo, df, va, te)
    else:
        print(f"'{SYNTH_DIR}' not found -> synthetic modes skipped")
        Xs, ys = np.zeros((0, N_SAMPLES), np.float32), np.zeros((0,), int)
    Ms = (mt.transform(prepare_meta(empty_meta(len(Xs)))).astype(np.float32)
          if len(Xs) else np.zeros((0, M_tr0.shape[1]), np.float32))
    print(f"Real train (after aug): {len(ytr_real)} | Synthetic kept: {len(ys)}")
    if len(ys):
        print("Synthetic class counts:", dict(zip(class_names, np.bincount(ys, minlength=n_classes))))

    results = []

    # ---- trivial baseline: always predict the most common training class
    maj = int(np.bincount(y[tr], minlength=n_classes).argmax())
    pb = np.full_like(yte, maj)
    bm = metrics_of(yte, pb, n_classes)
    print(f"\nMAJORITY BASELINE ('{class_names[maj]}') -> test acc {bm['acc']:.4f} | macro-F1 {bm['f1']:.4f} | balanced acc {bm['bal_acc']:.4f}")
    results.append({"run": "baseline_majority", "seed": np.nan, "data_mode": "baseline", "features": "-", "meta": False,
                    **{f"test_{k}": v for k, v in bm.items()}})

    for mode in DATA_MODES:
        if mode != "real" and len(Xs) == 0:
            print(f"Skipping '{mode}': no usable synthetic data"); continue
        if mode == "real":
            Xt, yt, Mt = Xtr_real, ytr_real, Mtr_real
        elif mode == "real+synth":
            Xt = np.concatenate([Xtr_real, Xs]); yt = np.concatenate([ytr_real, ys]); Mt = np.concatenate([Mtr_real, Ms])
        else:
            Xt, yt, Mt = Xs, ys, Ms

        for fs_name, kinds in FEATURE_SETS.items():
            Fe = build_features(kinds, Xt, Xva, Xte)
            for use_meta in USE_META_OPTIONS:
                if use_meta and mode == "synth_only":
                    continue
                for seed in SEEDS:
                    tag = f"{mode.replace('+', '_')}__{fs_name}__{'audio_meta' if use_meta else 'audio_only'}__s{seed}"
                    print(f"\n################ {tag} ################")
                    D = lambda i, M, yy: ([Fe[k][i] for k in kinds], M if use_meta else None, yy)
                    r = train_and_evaluate(kinds, n_classes, class_names,
                                           D(0, Mt, yt), D(1, M_va, yva), D(2, M_te, yte), tag, seed)
                    r.update({"data_mode": mode, "features": fs_name, "meta": use_meta})
                    results.append(r)
                    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "all_runs.csv"), index=False)
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
            del Fe

    comp = pd.DataFrame(results)
    cols = ["test_acc", "test_bal_acc", "test_f1", "val_f1"]
    summ = comp.groupby(["data_mode", "features", "meta"], dropna=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(OUT_DIR, "summary_mean_std.csv"), index=False)
    print(f"\n===== SUMMARY (mean/std over seeds {SEEDS}, test set is real data only) =====")
    print(summ.round(4).to_string(index=False))