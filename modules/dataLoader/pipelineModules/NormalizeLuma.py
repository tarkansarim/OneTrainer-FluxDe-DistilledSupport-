import torch
from typing import Optional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class NormalizeLuma(PipelineModule, RandomAccessPipelineModule):
	def __init__(
		self,
		image_in_name: str = "image",
		image_out_name: Optional[str] = None,
		mask_in_name: Optional[str] = None,
	):
		super().__init__()
		self.image_in_name = image_in_name
		self.image_out_name = image_out_name or image_in_name
		self.mask_in_name = mask_in_name

	def length(self) -> int:
		# Pass-through length; rely on upstream image stream
		return self._get_previous_length(self.image_in_name)

	def get_inputs(self) -> list[str]:
		names = [self.image_in_name, "concept"]
		if self.mask_in_name:
			names.append(self.mask_in_name)
		return names

	def get_outputs(self) -> list[str]:
		return [self.image_out_name]

	@staticmethod
	def _compute_luma(img: torch.Tensor) -> torch.Tensor:
		# img: (C,H,W) in [0,1]; assume RGB
		r, g, b = img[0], img[1], img[2]
		return 0.2126 * r + 0.7152 * g + 0.0722 * b

	@torch.no_grad()
	def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
		# Pull upstream items
		x: torch.Tensor = self._get_previous_item(variation, self.image_in_name, index)
		concept = self._get_previous_item(variation, "concept", index)
		mask = self._get_previous_item(variation, self.mask_in_name, index) if self.mask_in_name else None

		# Pass-through if inputs missing or disabled
		if not isinstance(x, torch.Tensor) or x.ndim < 3 or x.shape[-3] < 3 or not isinstance(concept, dict):
			return {self.image_out_name: x}
		image_cfg = (concept.get("image") or {}) if isinstance(concept.get("image"), dict) else {}
		if not bool(image_cfg.get("enable_luma_normalize", False)):
			return {self.image_out_name: x}

		target_mean = float(image_cfg.get("luma_target_mean", 0.5))
		target_std = float(image_cfg.get("luma_target_std", 0.25))
		mix = float(image_cfg.get("luma_mix", 1.0))
		mix = 0.0 if mix < 0.0 else (1.0 if mix > 1.0 else mix)

		# Ensure range [0,1]
		x = torch.clamp(x, 0.0, 1.0)

		# Mask if available
		if isinstance(mask, torch.Tensor) and mask.shape[-2:] == x.shape[-2:]:
			m = (mask > 0.5).to(x.dtype)
		else:
			m = None

		y = self._compute_luma(x)
		if m is not None:
			denom = m.sum().clamp_min(1.0)
			mu = (y * m).sum() / denom
			var = ((y - mu) ** 2 * m).sum() / denom
		else:
			mu = y.mean()
			var = y.var(unbiased=False)

		std = torch.sqrt(torch.clamp_min(var, 1e-6))
		scale = target_std / std
		shift = target_mean - mu * scale

		x_norm = x * scale + shift
		x_norm = torch.clamp(x_norm, 0.0, 1.0)

		if mix < 1.0:
			x_out = mix * x_norm + (1.0 - mix) * x
		else:
			x_out = x_norm

		return {self.image_out_name: x_out}

