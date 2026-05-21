import pytest
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from model_builder.connectors.audio import AudioConnector


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    """2 classes, 3 WAV files each (tiny sine waves)."""
    sr = 22050
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sr * duration))

    for cls, freq in [("speech", 440), ("music", 880)]:
        cls_dir = tmp_path / "audio" / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            wavfile.write(cls_dir / f"clip_{i}.wav", sr, wave)

    return tmp_path / "audio"


def test_connect_returns_connection(audio_dir):
    c = AudioConnector()
    conn = c.connect({"audio_dir": str(audio_dir), "n_mfcc": 5})
    assert conn.connector_name == "audio"


def test_connect_missing_dir_raises():
    c = AudioConnector()
    with pytest.raises(FileNotFoundError):
        c.connect({"audio_dir": "/nonexistent"})


def test_sample_returns_mfcc_features(audio_dir):
    c = AudioConnector()
    conn = c.connect({"audio_dir": str(audio_dir), "n_mfcc": 5, "label_from_dir": True})
    df = c.sample(conn, n=100)
    assert len(df) == 6
    assert "mfcc_0" in df.columns
    assert "mfcc_4" in df.columns
    assert "label" in df.columns
    assert set(df["label"].unique()) == {"speech", "music"}


def test_sample_respects_n(audio_dir):
    c = AudioConnector()
    conn = c.connect({"audio_dir": str(audio_dir), "n_mfcc": 5})
    df = c.sample(conn, n=2)
    assert len(df) <= 2


def test_profile_data_types(audio_dir):
    c = AudioConnector()
    conn = c.connect({"audio_dir": str(audio_dir), "n_mfcc": 5})
    profile = c.profile(conn)
    assert profile.row_count == 6
    assert "audio" in profile.data_types
    assert "tabular" in profile.data_types


def test_mfcc_column_count(audio_dir):
    c = AudioConnector()
    conn = c.connect({"audio_dir": str(audio_dir), "n_mfcc": 8, "label_from_dir": False})
    df = c.sample(conn, n=1)
    mfcc_cols = [col for col in df.columns if col.startswith("mfcc_")]
    assert len(mfcc_cols) == 8


async def test_stream_yields_chunks(audio_dir):
    c = AudioConnector()
    conn = c.connect({"audio_dir": str(audio_dir), "n_mfcc": 5,
                      "chunk_size": 2, "label_from_dir": True})
    chunks = []
    async for chunk in c.stream(conn):
        chunks.append(chunk)
    assert len(chunks) >= 1
    assert sum(len(ch.data) for ch in chunks) == 6
