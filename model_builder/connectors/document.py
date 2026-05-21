"""Document connector — extracts text from PDF, DOCX, TXT, and MD files."""
import glob as _glob
from pathlib import Path
from typing import AsyncIterator
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk

_DOC_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst"}


def _extract_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    rows = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            rows.append({
                "filename": path.name,
                "source": str(path),
                "page": page_num,
                "total_pages": len(reader.pages),
                "text": text,
                "char_count": len(text),
            })
    return rows


def _extract_docx(path: Path) -> list[dict]:
    from docx import Document
    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not full_text:
        return []
    return [{
        "filename": path.name,
        "source": str(path),
        "page": 1,
        "total_pages": 1,
        "text": full_text.strip(),
        "char_count": len(full_text),
    }]


def _extract_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [{
        "filename": path.name,
        "source": str(path),
        "page": 1,
        "total_pages": 1,
        "text": text,
        "char_count": len(text),
    }]


def _extract_file(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in (".docx", ".doc"):
        return _extract_docx(path)
    return _extract_text(path)


def _collect_docs(paths: list[str]) -> pd.DataFrame:
    rows = []
    for pattern in paths:
        matches = _glob.glob(pattern, recursive=True)
        targets = matches if matches else [pattern]
        for p in targets:
            path = Path(p)
            if path.is_file() and path.suffix.lower() in _DOC_EXTS:
                try:
                    rows.extend(_extract_file(path))
                except Exception as e:
                    rows.append({
                        "filename": path.name, "source": str(path),
                        "page": 1, "total_pages": 1,
                        "text": f"[ERROR: {e}]", "char_count": 0,
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["filename", "source", "page", "total_pages", "text", "char_count"]
    )


class DocumentConnector:
    name = "document"

    def connect(self, config: dict) -> Connection:
        paths = config.get("paths", [])
        if not paths:
            raise ValueError("DocumentConnector requires 'paths' in config")
        label_col = config.get("label_col")
        label_from_dir = config.get("label_from_dir", False)
        chunk_size = config.get("chunk_size", 500)
        return Connection(
            connector_name="document",
            handle={
                "paths": paths,
                "label_col": label_col,
                "label_from_dir": label_from_dir,
                "chunk_size": chunk_size,
            },
        )

    def _load(self, handle: dict) -> pd.DataFrame:
        df = _collect_docs(handle["paths"])
        if handle["label_from_dir"] and not df.empty:
            df["label"] = df["source"].apply(
                lambda p: Path(p).parent.name
            )
        return df

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        return self._load(conn.handle).head(n)

    def profile(self, conn: Connection) -> DataProfile:
        df = self._load(conn.handle)
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns={col: str(df[col].dtype) for col in df.columns},
            nulls={col: int(df[col].isna().sum()) for col in df.columns},
            data_types=["text"],
            sample_path=conn.handle["paths"][0] if conn.handle["paths"] else "",
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        df = self._load(conn.handle)
        chunk_size = conn.handle["chunk_size"]
        for i, start in enumerate(range(0, len(df), chunk_size)):
            yield Chunk(data=df.iloc[start: start + chunk_size], sequence=i)
