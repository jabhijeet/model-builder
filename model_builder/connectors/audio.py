"""Audio connector — reads WAV/MP3 files, extracts MFCC features via librosa."""
import asyncio
from pathlib import Path
from typing import AsyncIterator
import numpy as np
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _extract_features(path: Path, sr: int, n_mfcc: int) -> dict:
    import librosa
    import numpy as _np
    suffix = str(path).lower()
    if suffix.endswith((".wav", ".flac")):
        # soundfile avoids audioread's deprecated audioop module
        import soundfile as sf
        data, sr_actual = sf.read(str(path), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = data.astype(_np.float32)
        if sr is not None and sr != sr_actual:
            y = librosa.resample(y, orig_sr=sr_actual, target_sr=sr)
            sr_actual = sr
    else:
        y, sr_actual = librosa.load(str(path), sr=sr, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr_actual, n_mfcc=n_mfcc).mean(axis=1)
    return {f"mfcc_{i}": float(v) for i, v in enumerate(mfccs)}


def _collect_audio(audio_dir: Path, label_from_dir: bool, sr: int, n_mfcc: int) -> pd.DataFrame:
    records = []
    for f in sorted(audio_dir.rglob("*")):
        if f.suffix.lower() not in _AUDIO_EXTS:
            continue
        label = f.parent.name if label_from_dir else None
        try:
            features = _extract_features(f, sr, n_mfcc)
        except Exception:
            features = {f"mfcc_{i}": float("nan") for i in range(n_mfcc)}
        row = {"audio_path": str(f), "filename": f.name}
        if label_from_dir:
            row["label"] = label
        row.update(features)
        records.append(row)
    return pd.DataFrame(records)


class AudioConnector:
    name = "audio"

    def connect(self, config: dict) -> Connection:
        audio_dir = Path(config["audio_dir"])
        if not audio_dir.exists():
            raise FileNotFoundError(f"audio_dir not found: {audio_dir}")
        return Connection(
            connector_name="audio",
            handle={
                "audio_dir": str(audio_dir),
                "label_from_dir": config.get("label_from_dir", True),
                "sample_rate": config.get("sample_rate", 22050),
                "n_mfcc": config.get("n_mfcc", 13),
                "chunk_size": config.get("chunk_size", 50),
            },
        )

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        h = conn.handle
        return _collect_audio(
            Path(h["audio_dir"]), h["label_from_dir"], h["sample_rate"], h["n_mfcc"]
        ).head(n)

    def profile(self, conn: Connection) -> DataProfile:
        h = conn.handle
        df = _collect_audio(
            Path(h["audio_dir"]), h["label_from_dir"], h["sample_rate"], h["n_mfcc"]
        )
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns={col: str(df[col].dtype) for col in df.columns},
            nulls={col: int(df[col].isna().sum()) for col in df.columns},
            data_types=["audio", "tabular"],
            sample_path=h["audio_dir"],
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        h = conn.handle
        df = _collect_audio(
            Path(h["audio_dir"]), h["label_from_dir"], h["sample_rate"], h["n_mfcc"]
        )
        chunk_size = h["chunk_size"]
        for i, start in enumerate(range(0, len(df), chunk_size)):
            yield Chunk(data=df.iloc[start: start + chunk_size], sequence=i)
