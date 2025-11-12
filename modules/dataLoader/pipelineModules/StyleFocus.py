import torch
import torch.nn.functional as F
from typing import Optional

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class StyleFocus(PipelineModule, RandomAccessPipelineModule):
    """
    Emphasize style (shading/lighting/colors) by suppressing high-frequency structure.
    Modes:
      - downup: downsample then upsample full RGB (simple low-pass)
      - lowpass_luma: low-pass only luminance and remap RGB to match target luma
      - color_only: flatten luminance to a constant mean (palette focus)
    """
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
        return self._get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        names = [self.image_in_name, "concept"]
        if self.mask_in_name:
            names.append(self.mask_in_name)
        return names

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    @staticmethod
    def _luma(img: torch.Tensor) -> torch.Tensor:
        # img: (C,H,W) in [0,1], RGB
        r, g, b = img[0], img[1], img[2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @staticmethod
    def _downup(img: torch.Tensor, target_short_side: int) -> torch.Tensor:
        # img: (C,H,W) float32 [0,1]
        c, h, w = img.shape
        short = min(h, w)
        if short <= target_short_side or target_short_side <= 1:
            return img
        if w < h:
            new_w = target_short_side
            new_h = max(1, int(round(h / w * target_short_side)))
        else:
            new_h = target_short_side
            new_w = max(1, int(round(w / h * target_short_side)))
        x = img.unsqueeze(0)
        low = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False, antialias=True)
        up = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
        return up.squeeze(0)

    @torch.no_grad()
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        x: torch.Tensor = self._get_previous_item(variation, self.image_in_name, index)
        concept = self._get_previous_item(variation, "concept", index)
        if not isinstance(x, torch.Tensor) or x.ndim < 3 or x.shape[-3] < 3 or not isinstance(concept, dict):
            return {self.image_out_name: x}

        cfg = concept.get("image") or {}
        if not bool(cfg.get("enable_style_focus", False)):
            return {self.image_out_name: x}

        mode = str(cfg.get("style_focus_mode", "downup")).lower()
        short_side = int(cfg.get("style_focus_short_side", 128) or 128)
        blur_sigma = float(cfg.get("style_focus_blur_sigma", 3.0) or 3.0)  # reserved for future gaussian path
        mix = float(cfg.get("style_focus_mix", 0.7) or 0.7)
        mix = 0.0 if mix < 0.0 else (1.0 if mix > 1.0 else mix)
        color_mean = float(cfg.get("style_focus_color_mean", 0.5) or 0.5)

        x = torch.clamp(x, 0.0, 1.0)

        if mode == "downup":
            x_low = self._downup(x, short_side)
            out = mix * x_low + (1.0 - mix) * x
        elif mode == "lowpass_luma":
            y = self._luma(x).unsqueeze(0)  # (1,H,W)
            y_low = self._downup(y, short_side).squeeze(0)
            s = (y_low / (y.squeeze(0) + 1e-6)).clamp(0.0, 10.0)  # scale luminance
            out = (x * s.unsqueeze(0)).clamp(0.0, 1.0)
            out = mix * out + (1.0 - mix) * x
        elif mode == "color_only":
            y = self._luma(x)
            s = (color_mean / (y + 1e-6)).clamp(0.0, 10.0)
            out = (x * s.unsqueeze(0)).clamp(0.0, 1.0)
            out = mix * out + (1.0 - mix) * x
        else:
            out = x

        return {self.image_out_name: out}

