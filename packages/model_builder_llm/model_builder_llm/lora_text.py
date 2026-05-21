"""LoRA fine-tuning plugin for text classification."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_builder.plugins.protocol import (
    DataProfile, DataBundle, Artifact, EvalReport, TrainCallbacks, ExportFormat
)
from ._base import get_device, compute_metrics, get_lora_config, detect_target_modules


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]


class LoRATextPlugin:
    name = "lora_text"
    data_types = ["text"]
    requires_gpu = False

    def detect(self, profile: DataProfile) -> float:
        return 0.90 if "text" in profile.data_types else 0.0

    def train(self, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import get_peft_model

        df = pd.read_parquet(data.data_path)
        text_col = config.get("text_col") or _find_text_col(df)
        label_col = config.get("label_col") or df.columns[-1]

        texts = df[text_col].astype(str).tolist()
        raw_labels = df[label_col].tolist()

        # Encode labels to 0..N-1
        unique_labels = sorted(set(raw_labels))
        label2id = {l: i for i, l in enumerate(unique_labels)}
        labels = [label2id[l] for l in raw_labels]
        num_labels = len(unique_labels)

        base_model = config.get("base_model", "gpt2")
        max_len = config.get("max_length", 128)
        epochs = config.get("epochs", 3)
        batch_size = config.get("batch_size", 8)
        lr = config.get("learning_rate", 2e-4)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 16)
        device = get_device()

        callbacks.on_log(f"LoRA: base={base_model}, {len(texts)} samples, {num_labels} labels, device={device}")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        lora_cfg = get_lora_config(
            r=lora_r, alpha=lora_alpha,
            target_modules=detect_target_modules(base_model),
        )
        model = get_peft_model(model, lora_cfg)
        model.to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        callbacks.on_log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        dataset = _TextDataset(texts, labels, tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_enc, batch_labels in loader:
                batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                out = model(**batch_enc, labels=batch_labels)
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()
            avg = total_loss / max(len(loader), 1)
            callbacks.on_progress(epoch + 1, epochs, {"loss": round(avg, 4)})
            callbacks.on_log(f"Epoch {epoch+1}/{epochs} loss={avg:.4f}")

        # Save adapter weights + metadata
        model_dir = Path(data.data_path).parent / "lora_model"
        model_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

        meta = {
            "base_model": base_model,
            "text_col": text_col,
            "label_col": label_col,
            "label2id": label2id,
            "id2label": {str(v): k for k, v in label2id.items()},
            "num_labels": num_labels,
            "max_length": max_len,
        }
        (model_dir / "mb_meta.json").write_text(json.dumps(meta, indent=2))

        # Save a thin artifact pointer (not the full model)
        artifact_path = model_dir / "artifact.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump({"model_dir": str(model_dir), "meta": meta}, f)

        return Artifact(
            model_path=str(artifact_path),
            plugin_name="lora_text",
            metadata=meta,
        )

    def evaluate(self, artifact: Artifact, data: DataBundle) -> EvalReport:
        from transformers import AutoTokenizer
        from peft import PeftModel, PeftConfig, AutoPeftModelForSequenceClassification

        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)
        model_dir = saved["model_dir"]
        meta = saved["meta"]
        device = get_device()

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        model.eval()

        df = pd.read_parquet(data.data_path)
        texts = df[meta["text_col"]].astype(str).tolist()
        raw_labels = df[meta["label_col"]].tolist()
        label2id = meta["label2id"]
        y_true = np.array([label2id[l] for l in raw_labels])

        dataset = _TextDataset(texts, list(y_true), tokenizer, meta["max_length"])
        loader = DataLoader(dataset, batch_size=16)

        all_preds = []
        with torch.no_grad():
            for batch_enc, _ in loader:
                batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
                logits = model(**batch_enc).logits
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

        y_pred = np.array(all_preds)
        metrics = compute_metrics(y_true, y_pred)

        return EvalReport(plugin_name="lora_text", metrics=metrics)

    def export(self, artifact: Artifact, fmt: ExportFormat) -> Path:
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)
        model_dir = Path(saved["model_dir"])

        if str(fmt) == "safetensors":
            # Already saved as safetensors by save_pretrained
            st = model_dir / "adapter_model.safetensors"
            if st.exists():
                return st
        return model_dir / "artifact.pkl"


def _find_text_col(df: pd.DataFrame) -> str:
    """Heuristic: find the column most likely to contain text."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(5).astype(str)
            avg_len = sample.str.len().mean()
            if avg_len > 20:
                return col
    return df.select_dtypes(include=['object', 'string']).columns[0]
