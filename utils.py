import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def resolve_h5_dataset_path(name: str, cache_dir=None) -> Path:
    """Find ``<name>.h5`` under STABLEWM_HOME in either layout.

    Two swm versions disagree on the on-disk layout:

    * **0.0.6 wheel**: ``<STABLEWM_HOME>/<name>.h5`` (flat).
    * **Post-PR-#221 source**: ``<STABLEWM_HOME>/datasets/<name>.h5``
      (hard-coded ``sub_folder='datasets'``).

    This helper checks both candidate paths and returns whichever exists,
    so train.py / train_pldm.py can pass ``path=`` to ``HDF5Dataset`` and
    bypass the hard-coded sub_folder logic in the source-overlay version
    while still finding 0.0.6-style flat layouts.
    """
    base = Path(cache_dir) if cache_dir else Path(
        os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")
    )
    candidates = [
        base / f"{name}.h5",                  # 0.0.6 wheel / flat
        base / "datasets" / f"{name}.h5",      # post-PR-#221 layout
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"HDF5 dataset '{name}' not found; tried: {[str(p) for p in candidates]}"
    )


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


class AddGaussianBlur:
    """Per-frame Gaussian spatial blur on (..., C, H, W) tensors.

    Mirrors :class:`AddNormalizedGaussianNoise`'s API so the same
    eval / corruption-sweep machinery can dispatch on it:

    - ``Bernoulli(apply_prob)`` decides whether each frame is blurred;
    - if so, ``sigma ~ Uniform(sigma_min, sigma_max)`` is sampled per
      frame.

    Eval convention is ``sigma_min == sigma_max`` with ``apply_prob = 1.0``
    (deterministic blur of every frame).

    The kernel size defaults to the smallest odd integer
    :math:`\\geq 6 \\sigma + 1` so the support covers about
    :math:`\\pm 3 \\sigma` of the kernel mass.
    """

    def __init__(self, sigma_min, sigma_max, apply_prob: float = 1.0,
                 kernel_size: int | None = None):
        self.sigma_low = float(sigma_min)
        self.sigma_high = float(sigma_max)
        self.apply_prob = float(apply_prob)
        self.kernel_size = kernel_size
        if self.sigma_low < 0 or self.sigma_high < 0:
            raise ValueError("blur sigma must be non-negative")
        if self.sigma_low > self.sigma_high:
            raise ValueError(
                "sigma range must be ordered: "
                f"got sigma_min={sigma_min} > sigma_max={sigma_max}"
            )
        if not 0.0 <= self.apply_prob <= 1.0:
            raise ValueError(f"apply_prob must be in [0, 1], got {apply_prob}")

    @property
    def max_sigma(self) -> float:
        return self.sigma_high if self.apply_prob > 0 else 0.0

    @staticmethod
    def _kernel_size_for(sigma: float, override: int | None) -> int:
        if override is not None:
            return override if override % 2 == 1 else override + 1
        ks = 2 * int(math.ceil(3.0 * sigma)) + 1
        return max(ks, 3)

    def __call__(self, x):
        if not torch.is_tensor(x) or x.ndim < 3:
            return x
        if self.sigma_high <= 0 or self.apply_prob <= 0:
            return x

        # torchvision.transforms.v2 is available in this repo's deps
        from torchvision.transforms.v2.functional import gaussian_blur as _gblur

        # Fast path: deterministic per-frame sigma (eval convention).
        if self.sigma_low == self.sigma_high and self.apply_prob >= 1.0:
            sigma = self.sigma_high
            if sigma <= 0:
                return x
            ks = self._kernel_size_for(sigma, self.kernel_size)
            return _gblur(x, kernel_size=[ks, ks], sigma=[sigma, sigma])

        # Slow path: per-frame stochastic sigma (training convention).
        leading_shape = x.shape[:-3]
        n_frames = 1
        for d in leading_shape:
            n_frames *= int(d)
        x_flat = x.reshape(n_frames, *x.shape[-3:])
        sigmas = torch.empty(n_frames, device=x.device, dtype=x.dtype).uniform_(
            self.sigma_low, self.sigma_high
        )
        if self.apply_prob < 1.0:
            mask = (torch.rand(n_frames, device=x.device) < self.apply_prob).to(x.dtype)
            sigmas = sigmas * mask
        out = x_flat.clone()
        for i in range(n_frames):
            sigma = float(sigmas[i].item())
            if sigma <= 0:
                continue
            ks = self._kernel_size_for(sigma, self.kernel_size)
            out[i:i + 1] = _gblur(x_flat[i:i + 1], kernel_size=[ks, ks], sigma=[sigma, sigma])
        return out.reshape(*x.shape)


class AddResize:
    """Bilinear downscale-then-upscale: a low-pass that destroys
    high-frequency detail without adding noise.

    Per-frame independent sampling mirrors the noise transform's API:

    - ``Bernoulli(apply_prob)`` decides whether each frame is degraded;
    - if so, ``factor ~ Uniform(factor_min, factor_max)`` is sampled.

    ``factor = 1.0`` is a no-op; smaller factors discard more detail.
    The eval convention is ``factor_min == factor_max`` with
    ``apply_prob = 1.0``.
    """

    def __init__(self, factor_min, factor_max, apply_prob: float = 1.0):
        self.factor_low = float(factor_min)
        self.factor_high = float(factor_max)
        self.apply_prob = float(apply_prob)
        if not 0 < self.factor_low <= self.factor_high <= 1.0:
            raise ValueError(
                "resize factor must satisfy 0 < min <= max <= 1, "
                f"got ({factor_min}, {factor_max})"
            )
        if not 0.0 <= self.apply_prob <= 1.0:
            raise ValueError(f"apply_prob must be in [0, 1], got {apply_prob}")

    @property
    def max_resize_strength(self) -> float:
        """0 = no degradation, 1 = full destruction. Useful for diagnostic
        sensitivity sweeps that need a scalar 'amount of corruption'."""
        return (1.0 - self.factor_low) if self.apply_prob > 0 else 0.0

    @staticmethod
    def _resize_one(x_chw, factor: float):
        h, w = x_chw.shape[-2:]
        if factor >= 1.0:
            return x_chw
        h_low = max(1, int(round(h * factor)))
        w_low = max(1, int(round(w * factor)))
        small = F.interpolate(x_chw.unsqueeze(0), size=(h_low, w_low),
                              mode="bilinear", align_corners=False)
        big = F.interpolate(small, size=(h, w),
                            mode="bilinear", align_corners=False)
        return big.squeeze(0)

    def __call__(self, x):
        if not torch.is_tensor(x) or x.ndim < 3:
            return x
        if self.factor_low >= 1.0 or self.apply_prob <= 0:
            return x

        leading_shape = x.shape[:-3]
        n_frames = 1
        for d in leading_shape:
            n_frames *= int(d)
        x_flat = x.reshape(n_frames, *x.shape[-3:])

        # Fast path: deterministic factor (eval convention).
        if self.factor_low == self.factor_high and self.apply_prob >= 1.0:
            factor = self.factor_low
            out = torch.stack(
                [self._resize_one(x_flat[i], factor) for i in range(n_frames)],
                dim=0,
            )
            return out.reshape(*x.shape)

        # Slow path: per-frame stochastic factor.
        factors = torch.empty(n_frames, device=x.device, dtype=x.dtype).uniform_(
            self.factor_low, self.factor_high
        )
        if self.apply_prob < 1.0:
            mask = torch.rand(n_frames, device=x.device) < self.apply_prob
            factors = torch.where(mask, factors, torch.ones_like(factors))
        out = torch.stack(
            [self._resize_one(x_flat[i], float(factors[i].item())) for i in range(n_frames)],
            dim=0,
        )
        return out.reshape(*x.shape)


def build_eval_corruption(cfg):
    """Build an eval-time image-corruption transform from ``cfg.eval.corruption``.

    Dispatches on ``cfg.type`` (default ``gaussian``); returns ``None`` if
    the corruption is disabled or has a zero-magnitude parameter.

    Supported types and their parameters:

    - ``gaussian`` (default): additive ImageNet-space noise via
      :class:`AddNormalizedGaussianNoise`. Uses ``std``.
    - ``gaussian_blur``: spatial Gaussian blur via :class:`AddGaussianBlur`.
      Uses ``sigma``.
    - ``resize``: bilinear downscale-then-upscale via :class:`AddResize`.
      Uses ``factor``.
    """
    if cfg is None:
        return None
    ctype = _cfg_get(cfg, "type", "gaussian")
    if ctype == "gaussian":
        std = float(_cfg_get(cfg, "std", 0.0))
        if std <= 0:
            return None
        return AddNormalizedGaussianNoise(std, std)
    if ctype == "gaussian_blur":
        sigma = float(_cfg_get(cfg, "sigma", 0.0))
        if sigma <= 0:
            return None
        return AddGaussianBlur(sigma, sigma)
    if ctype == "resize":
        factor = float(_cfg_get(cfg, "factor", 1.0))
        if factor >= 1.0:
            return None
        return AddResize(factor, factor)
    raise ValueError(f"Unsupported corruption type: {ctype}")


def make_eval_corruption(magnitude: float, ctype: str = "gaussian"):
    """Build a corruption transform from a scalar magnitude and a type
    tag, intended for diagnostic-probe injection (where we want to
    parameterise the corruption strength as a single number rather than
    a config block).

    Magnitude semantics by type:
        gaussian      → noise std
        gaussian_blur → kernel sigma in pixels
        resize        → downscale-then-upscale factor (1.0 is a no-op)

    Returns ``None`` when the magnitude is the no-op value for the
    chosen type (0 for additive / blur; 1.0 for resize), so callers can
    short-circuit cleanly.
    """
    if ctype == "gaussian":
        if magnitude <= 0:
            return None
        return AddNormalizedGaussianNoise(magnitude, magnitude)
    if ctype == "gaussian_blur":
        if magnitude <= 0:
            return None
        return AddGaussianBlur(magnitude, magnitude)
    if ctype == "resize":
        if magnitude >= 1.0:
            return None
        return AddResize(magnitude, magnitude)
    raise ValueError(f"Unsupported corruption type: {ctype}")


def corruption_tag(cfg) -> str:
    """Build a filename-safe tag from a corruption config.

    Returns an empty string for an unconfigured or no-op corruption.
    Naming is chosen so blur/resize tags do not collide with the
    existing Gaussian-noise tag ``std<X>`` used throughout the eval
    summary tooling.

    - ``gaussian``      → ``std<X>``     (kept for backward compat)
    - ``gaussian_blur`` → ``blur_sigma<X>``
    - ``resize``        → ``rs_factor<X>``
    """
    if cfg is None:
        return ""
    ctype = _cfg_get(cfg, "type", "gaussian")
    if ctype == "gaussian":
        std = float(_cfg_get(cfg, "std", 0.0))
        if std <= 0:
            return ""
        return f"std{std:g}"
    if ctype == "gaussian_blur":
        sigma = float(_cfg_get(cfg, "sigma", 0.0))
        if sigma <= 0:
            return ""
        return f"blur_sigma{sigma:g}"
    if ctype == "resize":
        factor = float(_cfg_get(cfg, "factor", 1.0))
        if factor >= 1.0:
            return ""
        return f"rs_factor{factor:g}"
    raise ValueError(f"Unsupported corruption type: {ctype}")


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
