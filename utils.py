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


def _is_std_range(std) -> bool:
    return not isinstance(std, (str, bytes, int, float)) and hasattr(std, "__len__")


class AddNormalizedGaussianNoise:
    """Add Gaussian noise to ImageNet-normalized tensors using pixel-space std.

    `std` can be:
    - a scalar: fixed pixel-space noise level
    - a two-value sequence: sample uniformly from [low, high] on each call
    """

    def __init__(self, std):
        if _is_std_range(std):
            if len(std) != 2:
                raise ValueError(f"noise std range must have two values, got {std}")
            self.std_low = float(std[0])
            self.std_high = float(std[1])
        else:
            self.std_low = self.std_high = float(std)
        if self.std_low < 0 or self.std_high < 0:
            raise ValueError("noise std must be non-negative")
        if self.std_low > self.std_high:
            raise ValueError(
                f"noise std range must be ordered low <= high, got {std}"
            )
        stats = dt.dataset_stats.ImageNet
        channel_std = stats["std"] if isinstance(stats, dict) else stats.std
        self.channel_std = torch.as_tensor(channel_std, dtype=torch.float32)

    @property
    def max_std(self) -> float:
        return self.std_high

    def _sample_std(self) -> float:
        if self.std_low == self.std_high:
            return self.std_high
        return float(torch.empty(()).uniform_(self.std_low, self.std_high))

    def __call__(self, x):
        std = self._sample_std()
        if std <= 0:
            return x

        if not torch.is_tensor(x):
            return x
        if x.ndim < 3:
            return x + torch.randn_like(x) * std

        channel_dim = -3
        if x.shape[channel_dim] != self.channel_std.numel():
            return x + torch.randn_like(x) * std

        scale = (std / self.channel_std.to(device=x.device, dtype=x.dtype)).view(
            *([1] * (x.ndim - 3)), -1, 1, 1
        )
        return x + torch.randn_like(x) * scale


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_img_noise_transform(cfg, source: str = "pixels", target: str = "pixels"):
    noise_type = _cfg_get(cfg, "type", "gaussian")
    std = _cfg_get(cfg, "std", 0.0)
    noise = AddNormalizedGaussianNoise(std)

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

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

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

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
