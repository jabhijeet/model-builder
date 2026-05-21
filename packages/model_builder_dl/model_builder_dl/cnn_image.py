"""CNN plugin for image classification using torchvision ImageFolder."""
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from model_builder.plugins.protocol import (
    DataProfile, DataBundle, Artifact, EvalReport, TrainCallbacks, ExportFormat
)
from ._base import get_device


class _SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat = (img_size // 8) ** 2 * 128
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNImagePlugin:
    name = "cnn_image"
    data_types = ["image"]
    requires_gpu = False

    def detect(self, profile: DataProfile) -> float:
        return 0.90 if "image" in profile.data_types else 0.0

    def train(self, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact:
        image_dir = config.get("image_dir")
        if not image_dir:
            raise ValueError("CNNImagePlugin requires 'image_dir' in config")
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")

        img_size = config.get("img_size", 64)
        epochs = config.get("epochs", 5)
        batch_size = config.get("batch_size", 32)
        lr = config.get("learning_rate", 1e-3)
        device = get_device()

        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = datasets.ImageFolder(str(image_dir), transform=tfm)
        classes = dataset.classes
        num_classes = len(classes)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = _SimpleCNN(num_classes=num_classes, img_size=img_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        callbacks.on_log(f"CNNImage: {len(dataset)} images, {num_classes} classes, device={device}")

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(len(loader), 1)
            callbacks.on_progress(epoch + 1, epochs, {"loss": round(avg_loss, 4)})
            callbacks.on_log(f"Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

        model_dir = Path(data.data_path).parent / "cnn_model"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "state_dict": model.state_dict(),
                "classes": classes,
                "num_classes": num_classes,
                "img_size": img_size,
                "image_dir": str(image_dir),
            }, f)

        return Artifact(
            model_path=str(model_path),
            plugin_name="cnn_image",
            metadata={"classes": classes, "num_classes": num_classes, "img_size": img_size},
        )

    def evaluate(self, artifact: Artifact, data: DataBundle) -> EvalReport:
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)

        device = get_device()
        img_size = saved["img_size"]
        classes = saved["classes"]
        model = _SimpleCNN(num_classes=saved["num_classes"], img_size=img_size).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()

        image_dir = saved.get("image_dir", str(Path(artifact.model_path).parent.parent))
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = datasets.ImageFolder(image_dir, transform=tfm)
        loader = DataLoader(dataset, batch_size=64)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                preds = model(imgs.to(device)).argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        y_true, y_pred = np.array(all_labels), np.array(all_preds)
        avg = "binary" if saved["num_classes"] == 2 else "weighted"
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        }
        return EvalReport(plugin_name="cnn_image", metrics=metrics)

    def export(self, artifact: Artifact, fmt: ExportFormat) -> Path:
        model_dir = Path(artifact.model_path).parent
        out = model_dir / "model.pkl"
        return out  # already persisted as pkl
