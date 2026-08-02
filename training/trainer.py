"""Training manager module for Explainable AI Plant Disease Detection.

Implements two-stage transfer learning, Apple Silicon MPS channels_last optimization,
YOLO-style real-time training dashboard, EarlyStopping bug fix, and PyTorch 2.6+
primitive checkpoint serialization.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import TrainingConfig
from utils.metrics import calculate_classification_metrics

logger = logging.getLogger("trainer")


class EarlyStopping:
    """Early stopping handler based on validation score improvement."""

    def __init__(self, patience: int = 5, delta: float = 1e-4, mode: str = "max") -> None:
        """Initializes Early Stopping.

        Args:
            patience: Number of epochs to wait without improvement.
            delta: Minimum change to qualify as improvement.
            mode: 'max' for metrics like F1-score, 'min' for loss.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        """Updates counter and early_stop flag based on current score.

        Returns:
            bool: True if new best score achieved, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "max":
            improved = current_score > (self.best_score + self.delta)
        else:
            improved = current_score < (self.best_score - self.delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class TwoStageTrainer:
    """Manager for two-stage transfer learning with EfficientNet-B0."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ) -> None:
        """Initializes trainer with Apple Silicon MPS channels_last optimization."""
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Auto Device Detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = "NVIDIA CUDA GPU"
            self.use_amp = config.USE_AMP
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_name = "Apple MPS"
            self.use_amp = False  # CUDA-only amp scaler disabled on MPS
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"
            self.use_amp = False

        # Apply channels_last memory format optimization for CUDA, contiguous_format for MPS/CPU
        if self.device.type == "cuda":
            self.memory_format = torch.channels_last
        else:
            self.memory_format = torch.contiguous_format

        self.model = model.to(device=self.device, memory_format=self.memory_format)
        self.criterion = nn.CrossEntropyLoss()

        # Automatic Mixed Precision Scaler for CUDA
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "learning_rate": [],
        }

        logger.info("Device Selected: %s (AMP: %s)", self.device_name, self.use_amp)

    def _train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        stage_title: str,
        current_epoch: int,
    ) -> tuple[float, float]:
        """Runs one training epoch with live YOLO-style tqdm batch progress bar."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar_desc = f"[{stage_title}] Ep {current_epoch}/{self.config.MAX_TOTAL_EPOCHS}"
        pbar = tqdm(self.train_loader, desc=pbar_desc, leave=False, dynamic_ncols=True)

        for images, labels in pbar:
            # Transfer tensors to device with appropriate memory format
            images = images.to(self.device, memory_format=self.memory_format, non_blocking=False)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            current_loss = loss.item()
            current_acc = (correct / total) * 100.0
            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.1f}%"})

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    def _validate_one_epoch(self) -> tuple[float, float, float, float, float]:
        """Evaluates model on validation set."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, memory_format=self.memory_format, non_blocking=False)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = calculate_classification_metrics(np.array(all_targets), np.array(all_preds))

        return val_loss, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]

    def _save_checkpoints(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        best_f1: float,
        is_best: bool,
    ) -> None:
        """Saves PyTorch 2.6+ compliant primitive checkpoints in models/."""
        self.config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Pure primitive dictionary without non-serializable PosixPath objects
        checkpoint_state = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "best_f1_score": float(best_f1),
            "config": {
                "model_name": str(self.config.MODEL_NAME),
                "image_size": list(self.config.IMAGE_SIZE),
                "batch_size": int(self.config.BATCH_SIZE),
                "max_total_epochs": int(self.config.MAX_TOTAL_EPOCHS),
            },
        }

        # Save last model checkpoint
        last_path = self.config.MODELS_DIR / self.config.LAST_MODEL_NAME
        torch.save(checkpoint_state, last_path)

        # Save best model checkpoint
        if is_best:
            best_path = self.config.MODELS_DIR / self.config.BEST_MODEL_NAME
            torch.save(checkpoint_state, best_path)

        # Save optimizer state
        opt_path = self.config.MODELS_DIR / self.config.OPTIMIZER_STATE_FILE
        torch.save({"epoch": int(epoch), "optimizer_state_dict": optimizer.state_dict()}, opt_path)

        # Save training history JSON
        history_path = self.config.MODELS_DIR / self.config.HISTORY_FILE
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def print_epoch_summary(
        self,
        epoch: int,
        stage_name: str,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        precision: float,
        recall: float,
        f1_score: float,
        lr: float,
        epoch_time: float,
        elapsed_time: float,
        best_f1: float,
        checkpoint_saved: bool,
        early_stop_counter: int,
    ) -> None:
        """Prints a clean, formatted post-epoch dashboard box summary."""
        msg = (
            f"\n====================================================\n"
            f" Epoch {epoch} / {self.config.MAX_TOTAL_EPOCHS}\n"
            f" Stage                    : {stage_name}\n"
            f" Train Loss               : {train_loss:.4f}\n"
            f" Validation Loss          : {val_loss:.4f}\n"
            f" Train Accuracy           : {train_acc:.2f}%\n"
            f" Validation Accuracy      : {val_acc:.2f}%\n"
            f" Precision                : {precision:.2f}%\n"
            f" Recall                   : {recall:.2f}%\n"
            f" F1 Score                 : {f1_score:.2f}%\n"
            f" Learning Rate            : {lr:.2e}\n"
            f" Epoch Time               : {epoch_time:.2f}s\n"
            f" Elapsed Time             : {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n"
            f" Best F1                  : {best_f1:.2f}%\n"
            f" Checkpoint Saved         : {'Yes' if checkpoint_saved else 'No'}\n"
            f" EarlyStopping Counter    : {early_stop_counter} / {self.config.EARLY_STOPPING_PATIENCE}\n"
            f"====================================================\n"
        )
        logger.info(msg)

    def run_training(self) -> dict[str, Any]:
        """Executes two-stage training (Stage 1 Max 5 Epochs, Stage 2 Max 15 Epochs)."""
        pipeline_start = time.time()
        best_f1_score = 0.0
        current_epoch = 0

        # ==================== STAGE 1: Frozen Backbone ====================
        logger.info("Starting Stage 1: Frozen Backbone Training (Max %d Epochs)...", self.config.STAGE1_EPOCHS)
        self.model.freeze_backbone()

        optimizer_s1 = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.STAGE1_LR,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        scheduler_s1 = ReduceLROnPlateau(
            optimizer_s1, mode="min", factor=self.config.SCHEDULER_FACTOR, patience=self.config.SCHEDULER_PATIENCE
        )
        early_stopping_s1 = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE, mode="max")

        for ep in range(1, self.config.STAGE1_EPOCHS + 1):
            current_epoch += 1
            ep_start = time.time()
            current_lr = optimizer_s1.param_groups[0]["lr"]

            train_loss, train_acc = self._train_one_epoch(optimizer_s1, "Stage 1", current_epoch)
            val_loss, val_acc, precision, recall, f1 = self._validate_one_epoch()
            scheduler_s1.step(val_loss)

            ep_time = time.time() - ep_start
            elapsed_time = time.time() - pipeline_start

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["precision"].append(precision)
            self.history["recall"].append(recall)
            self.history["f1_score"].append(f1)
            self.history["learning_rate"].append(current_lr)

            is_best = early_stopping_s1(f1)
            if is_best:
                best_f1_score = f1

            self._save_checkpoints(current_epoch, optimizer_s1, best_f1_score, is_best)

            self.print_epoch_summary(
                epoch=current_epoch,
                stage_name="Stage 1 (Frozen Backbone)",
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                precision=precision,
                recall=recall,
                f1_score=f1,
                lr=current_lr,
                epoch_time=ep_time,
                elapsed_time=elapsed_time,
                best_f1=best_f1_score,
                checkpoint_saved=is_best,
                early_stop_counter=early_stopping_s1.counter,
            )

            # Check if EarlyStopping patience exceeded
            if early_stopping_s1.early_stop:
                logger.info("Stage 1 EarlyStopping triggered after %d epochs.", ep)
                break

        # ==================== STAGE 2: Fine-Tuning ====================
        stage2_max = min(self.config.STAGE2_EPOCHS, self.config.MAX_TOTAL_EPOCHS - current_epoch)
        if stage2_max > 0:
            logger.info("Starting Stage 2: Fine-Tuning (Max %d Epochs)...", stage2_max)
            self.model.unfreeze_final_blocks(unfreeze_from_block=6)

            optimizer_s2 = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.STAGE2_LR,
                weight_decay=self.config.WEIGHT_DECAY,
            )
            scheduler_s2 = ReduceLROnPlateau(
                optimizer_s2, mode="min", factor=self.config.SCHEDULER_FACTOR, patience=self.config.SCHEDULER_PATIENCE
            )
            early_stopping_s2 = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE, mode="max")

            for ep in range(1, stage2_max + 1):
                current_epoch += 1
                ep_start = time.time()
                current_lr = optimizer_s2.param_groups[0]["lr"]

                train_loss, train_acc = self._train_one_epoch(optimizer_s2, "Stage 2", current_epoch)
                val_loss, val_acc, precision, recall, f1 = self._validate_one_epoch()
                scheduler_s2.step(val_loss)

                ep_time = time.time() - ep_start
                elapsed_time = time.time() - pipeline_start

                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["precision"].append(precision)
                self.history["recall"].append(recall)
                self.history["f1_score"].append(f1)
                self.history["learning_rate"].append(current_lr)

                is_best = early_stopping_s2(f1)
                if is_best:
                    best_f1_score = f1

                self._save_checkpoints(current_epoch, optimizer_s2, best_f1_score, is_best)

                self.print_epoch_summary(
                    epoch=current_epoch,
                    stage_name="Stage 2 (Fine-Tuning)",
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    lr=current_lr,
                    epoch_time=ep_time,
                    elapsed_time=elapsed_time,
                    best_f1=best_f1_score,
                    checkpoint_saved=is_best,
                    early_stop_counter=early_stopping_s2.counter,
                )

                # Check if EarlyStopping patience exceeded
                if early_stopping_s2.early_stop:
                    logger.info("Stage 2 EarlyStopping triggered after %d fine-tuning epochs.", ep)
                    break

        total_elapsed = time.time() - pipeline_start

        return {
            "total_epochs": current_epoch,
            "best_f1_score": best_f1_score,
            "best_val_acc": max(self.history["val_acc"]) if self.history["val_acc"] else 0.0,
            "best_precision": max(self.history["precision"]) if self.history["precision"] else 0.0,
            "best_recall": max(self.history["recall"]) if self.history["recall"] else 0.0,
            "total_time_seconds": total_elapsed,
            "early_stopped": early_stopping_s1.early_stop or (early_stopping_s2.early_stop if 'early_stopping_s2' in locals() else False),
        }
