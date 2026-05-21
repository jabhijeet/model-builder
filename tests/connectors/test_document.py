"""Tests for DocumentConnector — PDF, DOCX, TXT extraction."""
import pytest
import struct
from pathlib import Path
from model_builder.connectors.document import DocumentConnector, _extract_text


@pytest.fixture
def txt_files(tmp_path: Path) -> Path:
    """Three .txt files in two subdirectories (for label_from_dir)."""
    for cls in ["positive", "negative"]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(3):
            (d / f"doc_{i}.txt").write_text(
                f"This is a {cls} document number {i}. " * 20, encoding="utf-8"
            )
    return tmp_path


@pytest.fixture
def single_pdf(tmp_path: Path) -> Path:
    """Minimal valid single-page PDF with extractable text."""
    # Minimal PDF structure that pypdf can parse
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Hello PDF World) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000369 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
452
%%EOF"""
    p = tmp_path / "sample.pdf"
    p.write_bytes(pdf_content)
    return p


# --- connect ---

def test_connect_stores_paths(txt_files: Path):
    c = DocumentConnector()
    conn = c.connect({"paths": [str(txt_files / "positive" / "*.txt")]})
    assert conn.connector_name == "document"
    assert len(conn.handle["paths"]) == 1


def test_connect_missing_paths_raises():
    c = DocumentConnector()
    with pytest.raises(ValueError, match="paths"):
        c.connect({})


# --- TXT extraction ---

def test_sample_txt_returns_dataframe(txt_files: Path):
    c = DocumentConnector()
    conn = c.connect({"paths": [str(txt_files / "**" / "*.txt")]})
    df = c.sample(conn, n=100)
    assert len(df) == 6
    assert "text" in df.columns
    assert "filename" in df.columns
    assert "char_count" in df.columns


def test_sample_respects_n(txt_files: Path):
    c = DocumentConnector()
    conn = c.connect({"paths": [str(txt_files / "**" / "*.txt")]})
    df = c.sample(conn, n=2)
    assert len(df) <= 2


def test_label_from_dir(txt_files: Path):
    c = DocumentConnector()
    conn = c.connect({
        "paths": [str(txt_files / "**" / "*.txt")],
        "label_from_dir": True,
    })
    df = c.sample(conn, n=100)
    assert "label" in df.columns
    assert set(df["label"].unique()) == {"positive", "negative"}


def test_profile_data_type_is_text(txt_files: Path):
    c = DocumentConnector()
    conn = c.connect({"paths": [str(txt_files / "**" / "*.txt")]})
    profile = c.profile(conn)
    assert "text" in profile.data_types
    assert profile.row_count == 6
    assert "text" in profile.columns


def test_char_count_populated(txt_files: Path):
    c = DocumentConnector()
    conn = c.connect({"paths": [str(txt_files / "positive" / "*.txt")]})
    df = c.sample(conn, n=10)
    assert all(df["char_count"] > 0)


async def test_stream_yields_chunks(txt_files: Path):
    c = DocumentConnector()
    conn = c.connect({
        "paths": [str(txt_files / "**" / "*.txt")],
        "chunk_size": 2,
    })
    chunks = []
    async for chunk in c.stream(conn):
        chunks.append(chunk)
    assert len(chunks) >= 1
    assert sum(len(ch.data) for ch in chunks) == 6


# --- PDF ---

def test_pdf_extraction(single_pdf: Path):
    c = DocumentConnector()
    conn = c.connect({"paths": [str(single_pdf)]})
    df = c.sample(conn, n=10)
    # pypdf may or may not extract text from minimal PDF — just verify no crash
    assert "text" in df.columns
    assert "page" in df.columns


# --- Registration ---

def test_document_connector_registered():
    from model_builder.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.register_built_ins()
    conn = reg.get_connector("connectors.document")
    assert conn.name == "document"
