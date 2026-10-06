import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)

# ------------------------------------------------------------------ CONFIG
METADATA_CSV = "combined_tracking_log.csv"     # <-- your CSV
DATA_DIR     = "Dataset"              # <-- the one big folder with data_00001.wav ...
FILENAME_FMT = "data_{:05d}.wav"             # data_00001.wav
INDEX_COL    = "Index"                       # column used to build the filename
INDEX_OFFSET = 0                             # set to 1 if Index=0 corresponds to data_00001

LABEL_COL    = "Diagnosis"                   # what the model predicts
GROUP_COLS   = ["Dataset", "Patient_ID"]     # used ONLY to keep a patient inside one split

# Metadata the model is allowed to use as input (NOT Patient_ID, NOT Index, NOT filename).
META_NUMERIC     = ["Age", "BMI", "Child Weight (kg)", "Child Height (cm)"]
META_CATEGORICAL = ["Gender", "Sound Type", "Recording Device", "Recording Mode", "Chest Location"]
# WARNING: Recording Device / Dataset-like columns can be a shortcut (e.g. most COPD clips may
# come from one device). That is why the script also trains an audio-only version to compare.

SR           = 16000
DURATION     = 6.0                           # seconds; check your median clip length
N_SAMPLES    = int(SR * DURATION)
N_MELS, N_FFT, HOP = 64, 1024, 512
SEED         = 42
EPOCHS       = 40
BATCH        = 32
AUG_TARGET_FRACTION = 0.5                    # minority classes augmented up to this x majority count
CACHE_DIR    = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED)


# ------------------------------------------------- 1. AUDIO STANDARDIZATION
def standardize(path):
    """Load -> resample -> mono -> fixed length -> normalize. Original file untouched."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)), mode="constant")
    else:
        start = (len(y) - N_SAMPLES) // 2
        y = y[start:start + N_SAMPLES]
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    return (y / peak if peak > 0 else y).astype(np.float32)


def load_or_build_dataset():
    x_path = os.path.join(CACHE_DIR, "X_wave.npy")
    m_path = os.path.join(CACHE_DIR, "meta.csv")
    if os.path.exists(x_path) and os.path.exists(m_path):
        return np.load(x_path), pd.read_csv(m_path)

    df = pd.read_csv(METADATA_CSV)
    df = df.dropna(subset=[LABEL_COL, INDEX_COL]).reset_index(drop=True)
    # df["filepath"] = df[INDEX_COL].apply(
    #     lambda i: os.path.join(DATA_DIR, FILENAME_FMT.format(int(i) + INDEX_OFFSET)))
    df["filepath"] = df[INDEX_COL].apply(
        lambda i: os.path.join(DATA_DIR, str(i))
    )

    missing = [p for p in df["filepath"] if not os.path.exists(p)]
    print(f"{len(df) - len(missing)}/{len(df)} audio files found.")
    if missing:
        print("First missing paths (check DATA_DIR / FILENAME_FMT / INDEX_OFFSET):", missing[:3])

    X, keep = [], []
    for i, row in df.iterrows():
        try:
            X.append(standardize(row["filepath"])); keep.append(i)
        except Exception as e:
            print(f"Skipping {row['filepath']}: {e}")
    df = df.loc[keep].reset_index(drop=True)
    X = np.stack(X)
    np.save(x_path, X); df.to_csv(m_path, index=False)
    return X, df


# ------------------------------------------------------- 2. TRAIN/VAL/TEST SPLIT
def split_data(y, df):
    """~72/14/14, stratified, and a patient never appears in two splits."""
    idx = np.arange(len(y))
    cols = [c for c in GROUP_COLS if c in df.columns]
    groups = df[cols].astype(str).agg("_".join, axis=1).values if cols else None

    def one_fold(indices, n_splits):
        if groups is not None:
            sp = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)
            return next(sp.split(indices, y[indices], groups[indices]))
        sp = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
        return next(sp.split(indices, y[indices]))

    rest, test = one_fold(idx, 7)
    rest, test = idx[rest], idx[test]
    tr, va = one_fold(rest, 6)
    return rest[tr], rest[va], test


# ------------------------------------------------------ 3. METADATA PREPROCESSING
def build_meta_transformer():
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, META_NUMERIC), ("cat", cat, META_CATEGORICAL)])


def prepare_meta(df):
    m = df[META_NUMERIC + META_CATEGORICAL].copy()
    for c in META_NUMERIC:
        m[c] = pd.to_numeric(m[c], errors="coerce")    # non-numeric -> NaN -> imputed
    for c in META_CATEGORICAL:
        m[c] = m[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return m


# ------------------------------------------------------------ 4. AUGMENTATION
def augment_wave(y):
    ops = np.random.choice(["noise", "shift", "pitch", "stretch"],
                           size=np.random.randint(1, 3), replace=False)
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
    peak = np.max(np.abs(y))
    return (y / peak if peak > 0 else y).astype(np.float32)


def augment_minority(X_tr, y_tr):
    """Training set only. Returns new X, y and `src` = index of the original row each sample
    came from, so augmented clips inherit the SAME metadata (age, gender, ...) as their source."""
    classes, counts = np.unique(y_tr, return_counts=True)
    target = int(counts.max() * AUG_TARGET_FRACTION)
    X_new, y_new, src_new = [], [], []
    for c, n in zip(classes, counts):
        if n >= target:
            continue
        pool = np.where(y_tr == c)[0]
        for _ in range(target - n):
            s = np.random.choice(pool)
            X_new.append(augment_wave(X_tr[s])); y_new.append(c); src_new.append(s)
    src = np.arange(len(y_tr))
    if X_new:
        X_tr = np.concatenate([X_tr, np.stack(X_new)])
        y_tr = np.concatenate([y_tr, np.array(y_new)])
        src = np.concatenate([src, np.array(src_new)])
    return X_tr, y_tr, src


# --------------------------------------------------------- 5. REPRESENTATIONS
def to_mel(X):
    out = []
    for y in X:
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        out.append(librosa.power_to_db(S, ref=np.max))
    return np.stack(out)[..., np.newaxis].astype(np.float32)


# ------------------------------------------------------------------- 6. MODELS
def _fuse_and_classify(audio_feat, inputs, meta_dim, n_classes, name):
    """Fully connected part. If meta_dim > 0, concatenate a metadata branch."""
    x = audio_feat
    if meta_dim > 0:
        m_in = layers.Input(shape=(meta_dim,), name="meta")
        m = layers.Dense(32, activation="relu")(m_in)
        m = layers.Dropout(0.2)(m)
        x = layers.Concatenate()([x, m])
        inputs = [inputs, m_in]
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, out, name=name)


def build_1d_cnn(input_len, n_classes, meta_dim=0):
    inp = layers.Input(shape=(input_len, 1), name="audio")
    x = layers.Conv1D(16, 64, strides=4, padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling1D(4)(x)
    for f in (32, 64, 128):
        x = layers.Conv1D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling1D(4)(x)
    x = layers.GlobalAveragePooling1D()(x)
    return _fuse_and_classify(x, inp, meta_dim, n_classes, "cnn_1d")


def build_2d_cnn(input_shape, n_classes, meta_dim=0):
    inp = layers.Input(shape=input_shape, name="audio")
    x = inp
    for f in (32, 64, 128):
        x = layers.Conv2D(f, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    return _fuse_and_classify(x, inp, meta_dim, n_classes, "cnn_2d")


# -------------------------------------------------- 7. TRAIN + 8. EVALUATION
def train_and_evaluate(model, Itr, ytr, Iva, yva, Ite, yte, class_names, tag):
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
           callbacks.ReduceLROnPlateau(patience=4, factor=0.5)]
    model.fit(Itr, ytr, validation_data=(Iva, yva), epochs=EPOCHS,
              batch_size=BATCH, class_weight=cw, callbacks=cbs, verbose=2)

    pred = np.argmax(model.predict(Ite, batch_size=BATCH), axis=1)
    print(f"\n===== {tag} =====")
    print(classification_report(yte, pred, labels=range(len(class_names)),
                                target_names=class_names, zero_division=0))
    cm = confusion_matrix(yte, pred, labels=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues"); plt.title(f"Confusion matrix - {tag}")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar(); plt.tight_layout()
    plt.savefig(f"cm_{tag}.png", dpi=150); plt.close()

    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    return {"model": tag, "accuracy": accuracy_score(yte, pred),
            "precision_macro": p, "recall_macro": r, "f1_macro": f1}


# ------------------------------------------------------------------------ MAIN
if __name__ == "__main__":
    X, df = load_or_build_dataset()
    le = LabelEncoder(); y = le.fit_transform(df[LABEL_COL].astype(str).str.strip())
    class_names = list(le.classes_)
    print("Class counts:", dict(zip(class_names, np.bincount(y))))

    tr, va, te = split_data(y, df)
    print(f"Split -> train {len(tr)}, val {len(va)}, test {len(te)}")

    # Metadata: fit on TRAIN rows only, then transform val/test
    meta_df = prepare_meta(df)
    mt = build_meta_transformer().fit(meta_df.iloc[tr])
    M_tr0, M_va, M_te = (mt.transform(meta_df.iloc[i]).astype(np.float32) for i in (tr, va, te))
    meta_dim = M_tr0.shape[1]
    print("Metadata feature dimension:", meta_dim)

    Xtr0, ytr0 = X[tr], y[tr]
    Xva, yva, Xte, yte = X[va], y[va], X[te], y[te]

    # Augmentation: TRAIN ONLY; augmented clips copy their source's metadata
    Xtr, ytr, src = augment_minority(Xtr0, ytr0)
    M_tr = M_tr0[src]
    print("Train after augmentation:", dict(zip(class_names, np.bincount(ytr))))

    # Representations
    wave = lambda a: a[..., np.newaxis]
    Mel_tr, Mel_va, Mel_te = to_mel(Xtr), to_mel(Xva), to_mel(Xte)
    mu, sd = Mel_tr.mean(), Mel_tr.std() + 1e-8            # train stats only
    Mel_tr, Mel_va, Mel_te = ((a - mu) / sd for a in (Mel_tr, Mel_va, Mel_te))

    representations = {
        "1D_waveform": (build_1d_cnn, N_SAMPLES, (wave(Xtr), wave(Xva), wave(Xte))),
        "2D_melspec":  (build_2d_cnn, Mel_tr.shape[1:], (Mel_tr, Mel_va, Mel_te)),
    }

    results = []
    for rep_name, (builder, shape, (A_tr, A_va, A_te)) in representations.items():
        # (a) audio only   (b) audio + metadata
        for use_meta in (False, True):
            tag = f"{rep_name}_{'audio+meta' if use_meta else 'audio_only'}".replace("+", "_")
            model = builder(shape, len(class_names), meta_dim if use_meta else 0)
            if use_meta:
                Itr, Iva, Ite = [A_tr, M_tr], [A_va, M_va], [A_te, M_te]
            else:
                Itr, Iva, Ite = A_tr, A_va, A_te
            results.append(train_and_evaluate(model, Itr, ytr, Iva, yva, Ite, yte,
                                              class_names, tag))
            tf.keras.backend.clear_session()

    comp = pd.DataFrame(results)
    print("\n", comp.round(4).to_string(index=False))
    comp.to_csv("model_comparison.csv", index=False)