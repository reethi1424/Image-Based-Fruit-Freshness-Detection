import os
import random
import zipfile
from pathlib import Path

import cv2
import gdown
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =============================
# Page setup
# =============================
st.set_page_config(
    page_title="Image-Based Fruit Freshness Detection",
    page_icon="🍎",
    layout="wide",
)

st.title("🍎 Image-Based Fruit Freshness Detection")
st.caption("Group 9 | Image-Based Fruit Freshness Detection")

st.markdown(
    """
    This application presents the results of a fruit freshness detection model
    using colour-based image features and a classical machine learning approach.
    """
)

st.divider()


# =============================
# Dataset download helpers (for Streamlit Cloud)
# =============================
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "dataset.zip"
DEFAULT_EXTRACT_DIR = DATA_DIR / "dataset"


def _extract_dir_candidates(base: Path) -> list[Path]:
    # We extract into data/, but zip content might create different top folder names.
    # Try common possibilities.
    return [
        base / "dataset",
        base / "archive" / "dataset" / "dataset",
        base / "dataset" / "dataset",
        base,
    ]


@st.cache_resource
def ensure_dataset_from_gdrive(file_id: str) -> Path | None:
    """
    Downloads dataset.zip from Google Drive into data/ and extracts it.
    Returns a detected dataset root folder that contains train/ and test/.
    """
    if not file_id:
        return None

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If already extracted and valid, return quickly
    for cand in _extract_dir_candidates(DATA_DIR):
        train_dir = cand / "train"
        test_dir = cand / "test"
        if train_dir.exists() and test_dir.exists():
            return cand

    # Download zip if missing
    if not ZIP_PATH.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(ZIP_PATH), quiet=False)

    # Extract zip
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(str(DATA_DIR))
    except Exception:
        return None

    # Detect root after extraction
    for cand in _extract_dir_candidates(DATA_DIR):
        train_dir = cand / "train"
        test_dir = cand / "test"
        if train_dir.exists() and test_dir.exists():
            return cand

    return None


# =============================
# Core functions (same style as notebook)
# =============================
def image_preprocessing_step(img_bgr, size=(224, 224)):
    if img_bgr is None:
        return None
    resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized, (3, 3), 0)


def get_fruit_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    mask = (s > 40).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def extract_hsv_features(img_rgb, use_mask=True):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    if use_mask:
        mask = get_fruit_mask(img_rgb)
        fruit_pixels = mask > 0
        H = h[fruit_pixels]
        S = s[fruit_pixels]
        V = v[fruit_pixels]
        if H.size == 0:
            H, S, V = h.flatten(), s.flatten(), v.flatten()
    else:
        H, S, V = h.flatten(), s.flatten(), v.flatten()

    return [
        float(np.mean(H)),
        float(np.std(H)),
        float(np.mean(S)),
        float(np.std(S)),
        float(np.mean(V)),
        float(np.std(V)),
    ]


def infer_label_from_path(path: Path):
    lower = str(path).lower()
    if "fresh" in lower:
        return 0
    if "rotten" in lower:
        return 1
    return None


def infer_fruit_from_path(path: Path):
    lower = str(path).lower()
    if "apple" in lower:
        return "Apple"
    if "banana" in lower:
        return "Banana"
    if "orange" in lower:
        return "Orange"
    return "Unknown"


def get_images_from_folder(folder: Path, limit=900, seed=42):
    random.seed(seed)
    items = []
    for p in folder.rglob("*.*"):
        label = infer_label_from_path(p)
        if label is None:
            continue
        fruit = infer_fruit_from_path(p)
        if fruit == "Unknown":
            continue
        items.append((p, label, fruit))
    random.shuffle(items)
    return items[:limit]


def build_feature_table(items, img_size=224, use_mask=True):
    rows = []
    for path, label, fruit in items:
        img_bgr = cv2.imread(str(path))
        img_bgr = image_preprocessing_step(img_bgr, size=(img_size, img_size))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        feats = extract_hsv_features(img_rgb, use_mask=use_mask)

        rows.append(
            {
                "H_mean": feats[0],
                "H_std": feats[1],
                "S_mean": feats[2],
                "S_std": feats[3],
                "V_mean": feats[4],
                "V_std": feats[5],
                "label": label,
                "fruit": fruit,
                "path": str(path),
            }
        )

    return pd.DataFrame(rows)


def train_svm(train_df, C=10.0):
    X = train_df[["H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std"]].values
    y = train_df["label"].values

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma="scale")),
        ]
    )
    model.fit(X, y)
    return model


def predict(model, df):
    X = df[["H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std"]].values
    return model.predict(X)


def plot_confusion_matrix(cm):
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    return fig


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Settings")

    st.subheader("Dataset")

    st.write("If you run locally, set a local dataset path.")
    st.write("If you deploy to Streamlit Cloud, use Google Drive zip download.")

    default_local_root = r"C:\Users\Athi\VIP PROJECT\archive\dataset\dataset"
    dataset_mode = st.radio("Dataset mode", ["Local folder", "Google Drive zip"], index=0)

    gdrive_file_id = ""
    dataset_root = ""

    if dataset_mode == "Local folder":
        dataset_root = st.text_input("Dataset root path", value=default_local_root)
        dataset_root_path = Path(dataset_root)
    else:
        st.info(
            "Upload dataset.zip to Google Drive, set access to anyone with link, then paste the file id here."
        )
        gdrive_file_id = st.text_input("Google Drive file id (dataset.zip)", value="19g_CpOxTeMt2YWxxLY9vx22oPgzuRz6o")
        dataset_root_path = ensure_dataset_from_gdrive(gdrive_file_id) or DEFAULT_EXTRACT_DIR

    train_dir = Path(dataset_root_path) / "train"
    test_dir = Path(dataset_root_path) / "test"

    st.write("Detected dataset root:", str(dataset_root_path))
    st.write("Train folder:", "✅ Found" if train_dir.exists() else "❌ Not found")
    st.write("Test folder:", "✅ Found" if test_dir.exists() else "❌ Not found")

    st.divider()
    st.subheader("Model options")
    img_size = st.selectbox("Image size", [224, 128, 64], index=0)
    use_mask = st.toggle("Use masking", value=True)
    C_val = st.number_input("SVM C", min_value=0.1, max_value=50.0, value=10.0, step=0.5)

    st.divider()
    st.subheader("Sampling")
    train_limit = st.number_input("Training samples", min_value=50, max_value=5000, value=900, step=50)
    test_limit = st.number_input("Testing samples", min_value=20, max_value=2000, value=300, step=20)

    st.divider()
    st.subheader("Gallery")
    gallery_n = st.number_input("Images to display", min_value=6, max_value=120, value=24, step=6)
    gallery_cols = st.selectbox("Columns", [2, 3, 4, 6], index=2)

    st.divider()
    st.subheader("Model save")
    model_path = st.text_input("Model file path", value=str(Path("models") / "svm_hsv_freshness.pkl"))


# =============================
# Main section
# =============================
st.subheader("Model Evaluation on Dataset")
st.write(
    "The model is evaluated using the prepared dataset. "
    "Results include classification performance metrics and visual prediction examples."
)

run = st.button("Run Evaluation", type="primary")

if run:
    if not train_dir.exists() or not test_dir.exists():
        st.error("Train or test folder not found. Please check dataset mode and dataset path settings.")
        st.stop()

    with st.spinner("Loading dataset images..."):
        train_items = get_images_from_folder(train_dir, limit=int(train_limit), seed=42)
        test_items = get_images_from_folder(test_dir, limit=int(test_limit), seed=42)

    if len(train_items) == 0 or len(test_items) == 0:
        st.error("No images found. Please confirm folder naming includes fresh or rotten and fruit names.")
        st.stop()

    with st.spinner("Extracting HSV features..."):
        train_df = build_feature_table(train_items, img_size=int(img_size), use_mask=bool(use_mask))
        test_df = build_feature_table(test_items, img_size=int(img_size), use_mask=bool(use_mask))

    if train_df.empty or test_df.empty:
        st.error("Feature extraction returned no data. Please verify image files in the dataset folders.")
        st.stop()

    with st.spinner("Training SVM and evaluating..."):
        model = train_svm(train_df, C=float(C_val))
        y_true = test_df["label"].values
        y_pred = predict(model, test_df)

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    out = test_df.copy()
    out["pred"] = y_pred
    out["true_name"] = out["label"].map({0: "Fresh", 1: "Rotten"})
    out["pred_name"] = out["pred"].map({0: "Fresh", 1: "Rotten"})
    out["correct"] = out["label"] == out["pred"]

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model}, model_path)

    st.success("Evaluation completed.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Accuracy", acc)
    c2.metric("Training Samples", int(len(train_df)))
    c3.metric("Testing Samples", int(len(test_df)))

    st.divider()

    st.subheader("Confusion Matrix")
    st.caption("Comparison between true labels and predicted labels")
    fig_cm = plot_confusion_matrix(cm)
    st.pyplot(fig_cm, clear_figure=True)

    st.subheader("Classification Report")
    st.caption("Precision, recall, F1-score, and support for each class")
    st.code(report)

    st.subheader("Prediction Visualisation")
    st.caption("Sample predictions from the test dataset")

    show_df = out.head(int(gallery_n))

    cols = st.columns(int(gallery_cols))
    for i, row in enumerate(show_df.itertuples(index=False)):
        col = cols[i % int(gallery_cols)]
        img_bgr = cv2.imread(row.path)
        img_bgr = image_preprocessing_step(img_bgr, size=(int(img_size), int(img_size)))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        status = "✅ Correct" if row.correct else "❌ Wrong"
        caption = f"{status}\nTrue: {row.true_name} | Pred: {row.pred_name}\nFruit: {row.fruit}"
        col.image(img_rgb, caption=caption, use_container_width=True)

    st.divider()
    st.caption("Academic demonstration using classical image processing and machine learning techniques")
else:
    st.info("Set the dataset mode and dataset path in the sidebar, then click Run Evaluation.")