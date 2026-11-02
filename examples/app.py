import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import pyarrow.parquet as pq
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL = "GlobalWheat/GWFSS_Segformer_b4"
DEFAULT_DATA = Path(__file__).parent / "train-00000-of-00001.parquet"
LABELS = ("background", "head", "stem", "leaf")
PALETTE = np.array([[0, 0, 0], [255, 190, 40], [50, 170, 240], [70, 200, 100]], dtype=np.uint8)

@st.cache_resource(show_spinner=False)
def load_model(device: str):
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(device).eval()
    return processor, model, Lock()

@torch.inference_mode()
def segment(image, processor, model, lock):
    device = next(model.parameters()).device
    with lock:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        inputs = processor(images=image, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        if logits.device != device:
            raise RuntimeError(f"Output on {logits.device}, expected {device}")
        logits = F.interpolate(
            logits, size=(image.height, image.width), mode="bilinear", align_corners=False
        )
        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report = {
            "model": MODEL,
            "model_device": str(device),
            "output_device": str(logits.device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "inference_seconds": round(time.perf_counter() - start, 3),
            "width": image.width,
            "height": image.height,
            "peak_torch_allocated_MiB": (
                round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                if device.type == "cuda" else None
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    return mask, report


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def preview(image):
    image = image.copy()
    image.thumbnail((900, 600))
    return image


st.set_page_config(page_title="Global Wheat Demo", layout="wide")
st.title("Global Wheat Demo")
st.caption("Demo Streamlit application for the Global Wheat Biohackathon 2026.")

with st.sidebar:
    st.header("Runtime")
    device = st.radio("Inference:", ["cuda", "cpu"], format_func=lambda d: (
        "CUDA" if d == "cuda" else "CPU"
    ))
    st.write({
        "Python": platform.python_version(),
        "PyTorch": str(torch.__version__),
        "PyTorch CUDA runtime": torch.version.cuda,
        "CUDA available": torch.cuda.is_available(),
        "Visible CUDA devices": torch.cuda.device_count(),
    })
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        st.write({
            "GPU": properties.name,
            "VRAM GiB": round(properties.total_memory / 2**30, 1),
            "Compute capability": f"{properties.major}.{properties.minor}",
        })

st.subheader("Sample data")
try:
    table = pq.read_table(DEFAULT_DATA, columns=["image", "image_id", "width", "height"])
    if table.num_rows == 0:
        raise ValueError("Error: empty Parquet file")
    image_ids = table["image_id"].to_pylist()
    row_index = st.selectbox("Image", range(table.num_rows), format_func=lambda i: image_ids[i])
    row = table.slice(row_index, 1).to_pylist()[0]
    image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
except Exception as exc:
    st.error(f"Cannot read sample data at {DEFAULT_DATA}: {exc}")
    st.stop()
st.caption(f"{table.num_rows} images · {image.width} × {image.height} pixels · {DEFAULT_DATA}")

if st.button("Load model"):
    try:
        with st.spinner("Loading model checkpoint..."):
            processor, model, model_lock = load_model(device)
        st.session_state["loaded_device"] = device
    except Exception as exc:
        st.session_state.pop("loaded_device", None)
        st.error(f"Model loading failed: {exc}")
loaded = st.session_state.get("loaded_device") == device 
result_key = (str(DEFAULT_DATA), DEFAULT_DATA.stat().st_mtime_ns, row_index, device)
if loaded:
    processor, model, model_lock = load_model(device)
else: 
    raise RuntimeError("Model could not be loaded.")

if st.button("Run inference", type="primary"):
    st.session_state.pop("prediction", None)
    try:
        with st.spinner(f"Running inference on {device}..."):
            mask, report = segment(image, processor, model, model_lock)
        report["image_id"] = row["image_id"]
        st.session_state["prediction"] = (result_key, mask, report)
    except Exception as exc:
        st.error(f"Inference failed on {device}; {exc}")

original_column, result_column = st.columns(2)
with original_column:
    st.subheader("Original")
    st.image(preview(image))

prediction = st.session_state.get("prediction")
with result_column:
    st.subheader("Segmentation")
    if prediction is not None and prediction[0] == result_key:
        _, mask, report = prediction
        opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
        color_mask = Image.fromarray(PALETTE[mask])
        overlay = Image.blend(image, color_mask, opacity)
        overlay = Image.composite(overlay, image, Image.fromarray((mask != 0).astype(np.uint8) * 255))
        st.image(preview(overlay))
        st.caption("Legend: \n  Gold: head\nBlue: stem\nGreen: leaf")
        st.download_button(
            "Download the mask (PNG)", png_bytes(Image.fromarray(mask)),
            file_name=f"{Path(row['image_id']).stem}-mask.png", mime="image/png",
        )

        _, mask, report = prediction
        st.json(report)
        counts = np.bincount(mask.ravel(), minlength=len(LABELS))
        st.table([
            {"Class ID": i, "Class": name, "Pixels": int(counts[i]),
            "Image %": round(100 * int(counts[i]) / mask.size, 2)}
            for i, name in enumerate(LABELS)
        ])


# NOTE(mateusz): unused
def test_nvidia_driver_passes():
    try:
        status = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}")


def test_cuda_found():
    device = "gpu" if torch.cuda.is_available() else "cpu" 
    if device == "gpu":
        try:
            with torch.inference_mode():
                x = torch.ones((128, 128), device="cuda:0")
                value = (x @ x)[0, 0].item()
                torch.cuda.synchronize()
                del x
            if value != 128:
                raise RuntimeError(f"CUDA matrix product returned {value}, expected 128")
        except Exception as exc:
            raise RuntimeError(f"CUDA operation failed: {exc}")

