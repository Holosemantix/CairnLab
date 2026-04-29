import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class AddNormalizedGaussianNoise:
    """Add Gaussian noise to ImageNet-normalized tensors using pixel-space std.

    Per-frame independent sampling (frames = leading dims before C,H,W):
    - Bernoulli(noise_prob) decides whether the frame gets noise
    - if yes, std ~ Uniform(std_min, std_max) is sampled per frame

    Backward compatible: std_min == std_max with noise_prob == 1.0 is the
    same as a fixed std applied to every frame.
    """

    def __init__(self, std_min, std_max, noise_prob: float = 1.0):
        self.std_low = float(std_min)
        self.std_high = float(std_max)
        self.noise_prob = float(noise_prob)
        if self.std_low < 0 or self.std_high < 0:
            raise ValueError("noise std must be non-negative")
        if self.std_low > self.std_high:
            raise ValueError(
                f"noise std range must be ordered std_min <= std_max, "
                f"got ({std_min}, {std_max})"
            )
        if not 0.0 <= self.noise_prob <= 1.0:
            raise ValueError(
                f"noise_prob must be in [0, 1], got {noise_prob}"
            )
        stats = dt.dataset_stats.ImageNet
        channel_std = stats["std"] if isinstance(stats, dict) else stats.std
        self.channel_std = torch.as_tensor(channel_std, dtype=torch.float32)

    @property
    def max_std(self) -> float:
        return self.std_high if self.noise_prob > 0 else 0.0

    def _sample_per_frame_std(self, leading_shape, device, dtype):
        stds = torch.empty(leading_shape, device=device, dtype=dtype).uniform_(
            self.std_low, self.std_high
        )
        if self.noise_prob < 1.0:
            mask = (torch.rand(leading_shape, device=device) < self.noise_prob).to(dtype)
            stds = stds * mask
        return stds

    def __call__(self, x):
        if not torch.is_tensor(x):
            return x
        if self.std_high <= 0 or self.noise_prob <= 0:
            return x

        if x.ndim < 3:
            stds = self._sample_per_frame_std((), x.device, x.dtype)
            return x + torch.randn_like(x) * stds

        leading_shape = x.shape[:-3]  # frame dims before (C, H, W); may be empty
        stds = self._sample_per_frame_std(leading_shape, x.device, x.dtype)
        per_frame_scale = stds.view(*leading_shape, 1, 1, 1)

        if x.shape[-3] == self.channel_std.numel():
            channel_factor = (1.0 / self.channel_std.to(device=x.device, dtype=x.dtype)).view(
                *([1] * len(leading_shape)), -1, 1, 1
            )
            scale = per_frame_scale * channel_factor
        else:
            scale = per_frame_scale

        return x + torch.randn_like(x) * scale


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_img_noise_transform(cfg, source: str = "pixels", target: str = "pixels"):
    noise_type = _cfg_get(cfg, "type", "gaussian")
    std_min = _cfg_get(cfg, "std_min", 0.0)
    std_max = _cfg_get(cfg, "std_max", 0.0)
    noise_prob = _cfg_get(cfg, "noise_prob", 1.0)
    noise = AddNormalizedGaussianNoise(std_min, std_max, noise_prob=noise_prob)

    if noise.max_std <= 0:
        return None
    if noise_type != "gaussian":
        raise ValueError(f"Unsupported image noise type: {noise_type}")

    return dt.transforms.WrapTorchTransform(
        noise, source=source, target=target
    )


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1, meta=None):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.meta = meta or {}

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

            self._save_meta()

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")

    def _save_meta(self):
        if not self.meta:
            return
        meta_path = self.dirpath / "training_meta.json"
        try:
            import json
            with open(meta_path, "w") as f:
                json.dump(self.meta, f, indent=2)
        except Exception as e:
            print(f"Error saving training meta: {e}")
