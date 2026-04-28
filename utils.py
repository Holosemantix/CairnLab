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
    """Add Gaussian noise to ImageNet-normalized tensors using pixel-space std."""

    def __init__(self, std: float):
        self.std = float(std)
        stats = dt.dataset_stats.ImageNet
        channel_std = stats["std"] if isinstance(stats, dict) else stats.std
        self.channel_std = torch.as_tensor(channel_std, dtype=torch.float32)

    def __call__(self, x):
        if self.std <= 0:
            return x

        if not torch.is_tensor(x):
            return x
        if x.ndim < 3:
            return x + torch.randn_like(x) * self.std

        channel_dim = -3
        if x.shape[channel_dim] != self.channel_std.numel():
            return x + torch.randn_like(x) * self.std

        scale = (self.std / self.channel_std.to(device=x.device, dtype=x.dtype)).view(
            *([1] * (x.ndim - 3)), -1, 1, 1
        )
        return x + torch.randn_like(x) * scale


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_img_noise_transform(cfg, source: str = "pixels", target: str = "pixels"):
    enabled = bool(_cfg_get(cfg, "enabled", False))
    noise_type = _cfg_get(cfg, "type", "gaussian")
    std = float(_cfg_get(cfg, "std", 0.0))

    if not enabled or std <= 0:
        return None
    if noise_type != "gaussian":
        raise ValueError(f"Unsupported image noise type: {noise_type}")

    return dt.transforms.WrapTorchTransform(
        AddNormalizedGaussianNoise(std), source=source, target=target
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
