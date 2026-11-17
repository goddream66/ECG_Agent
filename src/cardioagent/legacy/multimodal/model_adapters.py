from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os
import re

import numpy as np
from scipy import signal

from cardioagent.common.signal_tools import resample


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-8)


def upstream_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream implementation: {path}")
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PretrainedModels:
    def __init__(self, root: Path):
        self.root = root

    @lru_cache(maxsize=3)
    def load(self, modality: str):
        import torch

        torch.set_num_threads(min(torch.get_num_threads(), 4))
        upstream = self.root / "upstream"
        if modality == "ecg":
            return load_ecgfounder(self.root)
        elif modality == "ppg":
            source = upstream / "anyppg"
            network = upstream_module("anyppg_net1d", source / "resnet1d.py")
            path = source / "anyppg_ckpt.pth"
            state = torch.load(path, map_location="cpu", weights_only=True)
            labels = []
            model = network.Net1D(
                in_channels=1, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 512],
                m_blocks_list=[2, 2, 2, 3, 3, 1], kernel_size=3, stride=2, groups_width=16,
                use_bn=True, use_do=True, verbose=False,
            )
        else:
            source = upstream / "cardiophon-release" / "heart_sound_classification"
            network = upstream_module("cardiophon_byola", source / "byol_a2" / "models.py")
            path = source / "checkpoints" / "pretrained_physionet_2022_4kHz.pth"
            state = torch.load(path, map_location="cpu", weights_only=True)
            labels = []
            model = network.AudioNTT2022(n_mels=64, d=3072)
        model.load_state_dict(state, strict=True)
        return model.eval(), labels, path

    def predict(self, modality: str, channel: str, values: np.ndarray, fs: float, artifact: Path) -> dict:
        import torch

        if not np.isfinite(values).all() or np.ptp(values) < 1e-10:
            return {"status": "insufficient", "reason": "Waveform is flat or non-finite"}
        if modality == "ecg" and channel != "I":
            return {"status": "unavailable", "reason": "This ECGFounder adapter is configured for lead I"}
        if modality in ("ecg", "ppg") and len(values) / fs < 10:
            return {"status": "insufficient", "reason": "This adapter requires a window of at least 10 seconds"}
        model, labels, path = self.load(modality)
        result = {
            "status": "ok", "model_checkpoint": str(path.relative_to(self.root)),
            "device": "cpu", "output_type": "embedding",
        }
        with torch.inference_mode():
            if modality == "ecg":
                rate = 500
                chunk = resample(values, fs, rate)[:5000]
                b, a = signal.iirnotch(50, 30, rate)
                chunk = signal.filtfilt(b, a, chunk)
                b, a = signal.butter(4, [0.67, 40], btype="bandpass", fs=rate)
                chunk = zscore(signal.filtfilt(b, a, chunk))
                tensor = torch.tensor(chunk, dtype=torch.float32)[None, None, :]
                logits, features = model(tensor)
                scores = logits.sigmoid().flatten()
                top = torch.topk(scores, min(5, scores.numel()))
                result.update(
                    output_type="embedding_and_uncalibrated_class_scores",
                    top_scores=[
                        {"label": labels[idx], "score": round(float(score), 6)}
                        for score, idx in zip(top.values, top.indices)
                    ],
                    input_duration_s=10,
                )
            elif modality == "ppg":
                rate = 125
                chunk = zscore(resample(values, fs, rate)[:1250])
                features = model(torch.tensor(chunk, dtype=torch.float32)[None, None, :])
                result["input_duration_s"] = 10
            else:
                from torchaudio.transforms import MelSpectrogram

                rate = 4000
                samples = resample(values, fs, rate)
                segment_size = 3800
                usable_samples = len(samples) // segment_size * segment_size
                segments = torch.tensor(samples[:usable_samples], dtype=torch.float32).reshape(-1, segment_size)
                mel = MelSpectrogram(
                    sample_rate=rate, n_fft=512, win_length=512, hop_length=160,
                    n_mels=64, f_min=60, f_max=2000, power=2,
                )(segments)
                logs = (mel + torch.finfo(torch.float32).eps).log()
                logs = (logs - logs.mean(dim=(1, 2), keepdim=True)) / (
                    logs.std(dim=(1, 2), keepdim=True, unbiased=False) + 1e-8
                )
                features = model(logs.unsqueeze(1)).mean(0, keepdim=True)
                result.update(input_duration_s=round(usable_samples / rate, 4), segments=len(segments))
        array = features.detach().cpu().numpy()
        if not np.isfinite(array).all():
            raise ValueError("Pretrained model produced non-finite features")
        np.save(artifact, array)
        result.update(
            input_rate_hz=rate,
            embedding_shape=list(array.shape),
            embedding_file=artifact.name,
        )
        return result







from cardioagent.ecg_agent.models.ecgfounder import load_ecgfounder, normalize_ecg_device, ecgfounder_execution, cached_ecgfounder_scorer
