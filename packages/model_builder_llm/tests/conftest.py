"""
Test fixtures for model-builder-llm.

Uses a tiny GPT-2 saved locally — no HuggingFace Hub download needed.
Vocab size matches real GPT-2 tokenizer (50257) so tokens don't go OOB.
"""
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from model_builder.plugins.protocol import DataProfile, DataBundle


@pytest.fixture(scope="session")
def tiny_gpt2_dir(tmp_path_factory) -> Path:
    """
    Create a tiny GPT-2 model + tokenizer locally for testing.
    Uses real GPT-2 vocab_size (50257) with tiny embed dim (64) so
    tokenizer works without downloading the full model.
    """
    from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer
    model_dir = tmp_path_factory.mktemp("tiny_gpt2")

    # vocab_size must match GPT-2 tokenizer (50257) to avoid OOB token ids
    cfg = GPT2Config(
        n_embd=64, n_layer=2, n_head=2,
        n_positions=128, n_ctx=128,
        vocab_size=50257,
        num_labels=2,
    )
    model = GPT2ForSequenceClassification(cfg)
    model.save_pretrained(str(model_dir))

    # Load real GPT-2 tokenizer (cached after first run; ~1MB vocab files)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.save_pretrained(str(model_dir))

    return model_dir


@pytest.fixture
def text_bundle(tmp_path: Path) -> DataBundle:
    texts = [
        "This product is great and works perfectly",
        "Terrible quality, broke after one day",
        "Amazing value for the price I paid",
        "Would not recommend this at all",
        "Excellent customer service and fast shipping",
        "Complete waste of money, very disappointed",
        "Love it, exactly what I needed",
        "Poor build quality, fell apart quickly",
        "Outstanding performance, highly recommend",
        "Not what I expected, very misleading",
    ] * 10  # 100 samples

    labels = (["positive", "negative"] * 5) * 10

    df = pd.DataFrame({"review": texts[:100], "sentiment": labels[:100]})
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path, index=False)

    profile = DataProfile(
        row_count=100, column_count=2,
        columns={"review": "object", "sentiment": "object"},
        nulls={"review": 0, "sentiment": 0},
        data_types=["text"],
        sample_path=str(data_path),
    )
    return DataBundle(profile=profile, data_path=str(data_path))
