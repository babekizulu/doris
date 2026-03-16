from __future__ import annotations

import io
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st


st.set_page_config(page_title="Audio Feature Exporter", layout="wide")
st.title("Audio Feature Exporter")
st.write("Convert WAV/FLAC files into CSV feature data.")


def load_audio(uploaded_file) -> tuple[np.ndarray, int]:
    """
    Load audio from an uploaded file-like object while preserving native sample rate.
    Mix stereo to mono for consistent feature extraction.
    """
    uploaded_file.seek(0)
    data, sr = sf.read(uploaded_file)

    # Convert stereo/multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Ensure float32
    data = data.astype(np.float32)
    return data, sr


def amplitude_to_db_safe(y: np.ndarray) -> np.ndarray:
    """
    Convert amplitude to dB using RMS over frames.
    """
    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    return db


def extract_summary_features(y: np.ndarray, sr: int, filename: str) -> pd.DataFrame:
    """
    Produce a single-row summary DataFrame for one file.
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_value = float(np.squeeze(tempo))

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # dB from RMS
    db = amplitude_to_db_safe(y)

    row = {
        "filename": filename,
        "sample_rate": sr,
        "duration_seconds": float(duration),
        "tempo_bpm": tempo_value,
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "db_mean": float(np.mean(db)),
        "db_std": float(np.std(db)),
    }

    for i in range(mfcc.shape[0]):
        row[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        row[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    for i in range(chroma.shape[0]):
        row[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        row[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    return pd.DataFrame([row])


def extract_frame_features(y: np.ndarray, sr: int) -> pd.DataFrame:
    """
    Produce frame-by-frame feature data.
    """
    hop_length = 512

    # Feature matrices
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    db = librosa.amplitude_to_db(rms, ref=np.max)

    # Ensure same frame count
    n_frames = min(
        mfcc.shape[1],
        chroma.shape[1],
        spectral_centroid.shape[1],
        db.shape[1],
    )

    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    data = {
        "time_seconds": times,
        "spectral_centroid": spectral_centroid[0, :n_frames],
        "db": db[0, :n_frames],
    }

    for i in range(13):
        data[f"mfcc_{i+1}"] = mfcc[i, :n_frames]

    for i in range(12):
        data[f"chroma_{i+1}"] = chroma[i, :n_frames]

    return pd.DataFrame(data)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_zip(file_map: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in file_map.items():
            zf.writestr(name, content)
    buffer.seek(0)
    return buffer.read()


mode = st.radio(
    "Export mode",
    ["Summary CSV (one row per file)", "Frame-by-frame CSV (one CSV per file)"],
    horizontal=True,
)

uploaded_files = st.file_uploader(
    "Upload WAV or FLAC files",
    type=["wav", "flac"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) loaded.")

    if mode == "Summary CSV (one row per file)":
        all_rows = []

        for uploaded_file in uploaded_files:
            try:
                y, sr = load_audio(uploaded_file)
                df = extract_summary_features(y, sr, uploaded_file.name)
                all_rows.append(df)
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")

        if all_rows:
            summary_df = pd.concat(all_rows, ignore_index=True)
            st.subheader("Preview")
            st.dataframe(summary_df, use_container_width=True)

            st.download_button(
                label="Download summary CSV",
                data=df_to_csv_bytes(summary_df),
                file_name="audio_features_summary.csv",
                mime="text/csv",
            )

    else:
        zip_contents: dict[str, bytes] = {}

        for uploaded_file in uploaded_files:
            try:
                y, sr = load_audio(uploaded_file)
                frame_df = extract_frame_features(y, sr)
                out_name = f"{Path(uploaded_file.name).stem}_frame_features.csv"
                zip_contents[out_name] = df_to_csv_bytes(frame_df)

                with st.expander(f"Preview: {uploaded_file.name}"):
                    st.dataframe(frame_df.head(20), use_container_width=True)

            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")

        if zip_contents:
            zip_bytes = build_zip(zip_contents)
            st.download_button(
                label="Download ZIP of frame-by-frame CSVs",
                data=zip_bytes,
                file_name="audio_frame_features.zip",
                mime="application/zip",
            )