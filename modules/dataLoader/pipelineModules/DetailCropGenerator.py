import hashlib
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import InterpolationMode, functional as F

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


@dataclass
class DetailEntry:
    base_index: int
    variant_type: str  # 'full', 'detail', 'context'
    scale: Optional[float]
    tile_index: int
    top: int
    left: int
    tile_size: int
    source_size: Optional[Tuple[int, int]]
    base_resolution: Tuple[int, int]


class DetailCropGenerator(PipelineModule, RandomAccessPipelineModule):
    """
    Pipeline module that duplicates high-resolution samples into deterministic detail crops.
    """

    def __init__(
            self,
            image_name: str,
            concept_name: str,
            image_path_name: str,
            additional_image_like_names: Optional[Dict[str, InterpolationMode]] = None,
            export_root: Optional[str] = None,
            passthrough_names: Optional[Iterable[str]] = None,
    ):
        self.image_name = image_name
        self.concept_name = concept_name
        self.image_path_name = image_path_name
        self.additional_image_like_names = additional_image_like_names or {}
        self.export_root = export_root
        self.passthrough_names = tuple(sorted(set(passthrough_names or [])))
        self._entries_by_base: Dict[int, List[DetailEntry]] = {}
        super().__init__()

        self._init_internal_state()

    def _init_internal_state(self):
        self._entries: List[DetailEntry] = []
        self._scale_remaining: Dict[Tuple[int, str, Optional[float]], int] = {}
        self._resized_cache: Dict[str, Dict[Tuple[int, str, Optional[float]], torch.Tensor]] = {self.image_name: {}}
        for name in self.additional_image_like_names:
            self._resized_cache[name] = {}
        self._export_limits: Dict[int, int] = {}
        self._export_counts: Dict[int, int] = {}
        self._export_limit_warnings: set[int] = set()
        self._export_dir_errors: set[str] = set()
        self._progress_prefix: str = ""
        self._progress_total: int = 0
        self._progress_count: int = 0
        self._progress_last_percent: int = -1

    @staticmethod
    def _concept_label(concept) -> str:
        if concept is None:
            return "unknown-concept"
        if isinstance(concept, dict):
            for key in ("name", "concept_name", "path", "concept_path"):
                value = concept.get(key)
                if value:
                    return str(value)
        for attr in ("name", "concept_name", "path", "concept_path"):
            value = getattr(concept, attr, None)
            if value:
                return str(value)
        return str(concept)

    @staticmethod
    def _normalize_path(path_value) -> str:
        if path_value is None:
            return "unknown-path"
        if isinstance(path_value, (list, tuple)):
            return ", ".join(str(item) for item in path_value)
        return str(path_value)

    @staticmethod
    def _debug(message: str):
        print(f"[Detail Crops Debug] {message}")
        sys.stdout.flush()

    def clear_item_cache(self):
        super().clear_item_cache()

    def length(self) -> int:
        # If entries haven't been populated yet (start() not called), populate them now
        if not self._entries:
            print(f"[Detail Crops] length() called with no entries, calling start(0) to populate...")
            sys.stdout.flush()
            self.start(0)  # Use variation 0 for length calculation
        result = len(self._entries)
        print(f"[Detail Crops] length() returning {result} entries")
        sys.stdout.flush()
        return result

    def get_inputs(self) -> List[str]:
        inputs = [self.image_name, self.concept_name, self.image_path_name]
        inputs.extend(self.additional_image_like_names.keys())
        inputs.extend(self.passthrough_names)
        return inputs

    def get_outputs(self) -> List[str]:
        outputs = [self.image_name, self.concept_name, self.image_path_name]
        outputs.extend(self.additional_image_like_names.keys())
        outputs.extend(self.passthrough_names)
        outputs.extend([
            'detail_crop_type',
            'detail_crop_scale',
            'detail_crop_index',
            'detail_crop_coords',
            'detail_crop_source_resolution',
            'detail_crop_prompt_context',
        ])
        return outputs

    def start(self, variation: int):
        super().clear_item_cache()
        self._init_internal_state()
        
        # Determine distributed context for optimized crop sharing
        try:
            import torch
            from modules.util import multi_gpu_util as _multi
            world_size = int(getattr(_multi, "world_size")())
            rank = int(getattr(_multi, "rank")())
            is_distributed = world_size > 1
        except Exception:
            world_size, rank, is_distributed = 1, 0, False

        # Master rank (or single-GPU) logic:
        # Calculate base length and check if any crops are enabled
        try:
            # Use image_path length to avoid triggering image decoding during length probing
            base_length = self._get_previous_length(self.image_path_name)
        except Exception as exc:
            self._debug(
                f"[Detail Crops] ERROR querying upstream length at variation {variation}: {exc}"
            )
            base_length = 0
        if base_length == 0:
            self._debug(
                "[Detail Crops] Upstream reported zero base samples; the dataset will be empty. "
                "Verify concept configuration, include_full_images, and blank-tile thresholds."
            )
        
        # Check if detail crops are enabled for any concept to determine verbosity and distributed need
        any_crops_enabled = False
        for base_index in range(min(base_length, 1)):  # Check first concept
            try:
                concept = self._get_previous_item(variation, self.concept_name, base_index)
                detail_cfg = self._extract_detail_config(concept)
                if detail_cfg.get('enabled', False) and detail_cfg.get('scales', []):
                    any_crops_enabled = True
                    break
            except Exception:
                pass
        
        if any_crops_enabled:
            # Get scales from first enabled concept for display
            scales_display = "base resolution"
            try:
                concept = self._get_previous_item(variation, self.concept_name, 0)
                detail_cfg = self._extract_detail_config(concept)
                scales = detail_cfg.get('scales', [])
                if scales:
                    scales_str = ", ".join(f"{s}x" for s in sorted(scales))
                    scales_display = f"for {scales_str} tiled upscale detail training"
                else:
                    scales_display = "at base resolution"
            except Exception:
                pass
            
            self._debug(f"[Detail Crops] Generating tiled crops for epoch {variation} {scales_display}")
            self._debug(f"[Detail Crops] Processing {base_length} base images")
            self._reset_progress(f"[Detail Crops] Epoch {variation}", base_length)
        
        # In multi-GPU mode, use cache ONLY if crops are enabled (to share generation work).
        # If disabled, we are just passing through full images, which is fast and deterministic locally.
        use_distributed_cache = is_distributed and any_crops_enabled
        cache_file_path = None

        if use_distributed_cache:
            import os
            import pickle
            cache_dir = os.path.join(os.getcwd(), "workspace-cache", "detail_crops_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file_path = os.path.join(cache_dir, f"crops_epoch_{variation}.pkl")
            
            if rank != 0:
                # Worker ranks: Wait for master to finish, then load crops from cache
                print(f"[Detail Crops] Rank {rank} waiting for master to generate crops...")
                sys.stdout.flush()
                torch.distributed.barrier()
                
                if os.path.exists(cache_file_path):
                    print(f"[Detail Crops] Rank {rank} loading crops from cache...")
                    sys.stdout.flush()
                    with open(cache_file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        self._entries = cached_data['entries']
                        self._entries_by_base = cached_data['entries_by_base']
                        self._scale_remaining = cached_data['scale_remaining']
                    print(f"[Detail Crops] Rank {rank} loaded {len(self._entries)} crops from cache")
                    sys.stdout.flush()
                    torch.distributed.barrier()
                    return
                else:
                    print(f"[Detail Crops] Rank {rank} ERROR: cache file not found at {cache_file_path}")
                    sys.stdout.flush()
                    torch.distributed.barrier()
                    raise RuntimeError(f"Detail crops cache file missing for Rank {rank}")
        
        # Local Generation Logic (Master rank OR all ranks if sync skipped)
        rebuild_all = not self._entries_by_base
        new_entries: List[DetailEntry] = []
        summary_counts: Counter[str] = Counter()
        for base_index in range(base_length):
            concept = self._get_previous_item(variation, self.concept_name, base_index)
            detail_cfg = self._extract_detail_config(concept)
            image_tensor = self._get_previous_item(variation, self.image_name, base_index)
            image_path_value = self._get_previous_item(variation, self.image_path_name, base_index)
            if image_tensor is None:
                self._debug(
                    f"Base index {base_index} ({self._normalize_path(image_path_value)}) has no image tensor; skipping."
                )
                continue

            base_resolution = (int(image_tensor.shape[-1]), int(image_tensor.shape[-2]))
            needs_rebuild = (
                rebuild_all
                or detail_cfg.get('regenerate_each_epoch', False)
                or base_index not in self._entries_by_base
            )

            variant_logs: List[dict] | None = None
            if needs_rebuild:
                entries, variant_logs = self._build_entries_for_image(
                    image_tensor,
                    detail_cfg,
                    base_index,
                    base_resolution,
                )
                include_full = detail_cfg.get('include_full_images', True)
                if not entries and include_full:
                    entries = [
                        DetailEntry(
                            base_index=base_index,
                            variant_type='full',
                            scale=None,
                            tile_index=0,
                            top=0,
                            left=0,
                            tile_size=base_resolution[0],
                            source_size=None,
                            base_resolution=base_resolution,
                        )
                    ]
                self._entries_by_base[base_index] = entries

            entries = self._entries_by_base.get(base_index, []).copy()

            # Only filter full images if detail crops are enabled.
            # If disabled, we must pass through the original image (full) to support standard training.
            if detail_cfg.get('enabled', False) and not detail_cfg.get('include_full_images', True):
                entries = [entry for entry in entries if entry.variant_type != 'full']
                self._entries_by_base[base_index] = entries

            if detail_cfg['save_to_disk']:
                if base_index not in self._export_limits:
                    self._export_limits[base_index] = detail_cfg['save_max_tiles_per_image']
                if base_index not in self._export_counts:
                    self._export_counts[base_index] = 0

            for entry in entries:
                if entry.variant_type != 'full':
                    key = (entry.base_index, entry.variant_type, entry.scale)
                    self._scale_remaining[key] = self._scale_remaining.get(key, 0) + 1

            new_entries.extend(entries)

            self._export_entries_to_disk(
                variation=variation,
                base_image=image_tensor,
                image_path=image_path_value,
                entries=entries,
                detail_cfg=detail_cfg,
            )
            summary_counts.update(entry.variant_type for entry in entries)

            detail_count = sum(1 for entry in entries if entry.variant_type == 'detail')
            context_count = sum(1 for entry in entries if entry.variant_type == 'context')
            full_count = sum(1 for entry in entries if entry.variant_type == 'full')
            
            # Only show debug messages if crops are actually being generated
            if any_crops_enabled and detail_count == 0 and context_count == 0:
                # ... existing debug logic ...
                pass

            if any_crops_enabled:
                self._increment_progress()

        self._entries = new_entries
        
        # In distributed mode, save crops to cache so other ranks can load them
        if use_distributed_cache and rank == 0 and cache_file_path:
            import pickle
            print(f"[Detail Crops] Rank {rank} saving crops to cache...")
            sys.stdout.flush()
            cache_data = {
                'entries': self._entries,
                'entries_by_base': self._entries_by_base,
                'scale_remaining': self._scale_remaining,
            }
            with open(cache_file_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[Detail Crops] Rank {rank} saved {len(self._entries)} crops to cache")
            sys.stdout.flush()
            # First barrier: Let Rank 1 know crops are ready to load
            torch.distributed.barrier()
            # Second barrier: Wait for Rank 1 to finish loading from cache
            torch.distributed.barrier()
        
        if any_crops_enabled:
            self._finalize_progress()
        
        detail_total = summary_counts.get('detail', 0)
        context_total = summary_counts.get('context', 0)
        full_total = summary_counts.get('full', 0)
        
        # Only show summary if crops are enabled
        if any_crops_enabled:
            self._debug(
                f"[Detail Crops] Epoch {variation} summary: "
                f"base_images={base_length}, detail_tiles={detail_total}, "
                f"context_tiles={context_total}, full_images={full_total}, total_entries={len(self._entries)}"
            )
            if len(self._entries) == 0:
                self._debug(
                    "[Detail Crops] WARNING: no crops generated for this variation. "
                    "Check tile scales, blank detection thresholds, and include_full_images settings."
                )
            elif detail_total == 0 and context_total == 0:
                self._debug(
                    "[Detail Crops] NOTE: only full images are available for this variation. "
                    "If detail crops are required, adjust the detail crop configuration."
                )

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        # Guard against rare out-of-range requests from upstream batchers
        total = len(self._entries)
        if total == 0:
            raise IndexError("DetailCropGenerator has no entries available for this variation.")
        if index < 0 or index >= total:
            index = index % total
        entry = self._entries[index]
        
        concept = self._get_previous_item(variation, self.concept_name, entry.base_index)
        detail_cfg = self._extract_detail_config(concept)

        base_image = self._get_previous_item(variation, self.image_name, entry.base_index)
        if base_image is None:
            raise RuntimeError("DetailCropGenerator depends on upstream image tensor, but none was provided.")

        image_path = self._get_previous_item(variation, self.image_path_name, entry.base_index)

        out_image: torch.Tensor
        if entry.variant_type == 'full':
            out_image = base_image
        else:
            resized = self._get_or_create_resized_tensor(
                entry,
                self.image_name,
                base_image,
                InterpolationMode.BILINEAR,
            )
            out_image = self._crop_tensor(resized, entry)

        output = {
            self.image_name: out_image,
            self.concept_name: concept,
            self.image_path_name: self._build_image_identifier(image_path, entry),
            'detail_crop_type': entry.variant_type,
            'detail_crop_scale': 1.0 if entry.scale is None else entry.scale,
            'detail_crop_index': entry.tile_index,
            'detail_crop_coords': (entry.top, entry.left),
            'detail_crop_source_resolution': entry.source_size if entry.source_size else entry.base_resolution,
        }

        original_prompt: Optional[str] = None
        for name in self.passthrough_names:
            value = self._get_previous_item(variation, name, entry.base_index)
            if name == "prompt":
                original_prompt = value
                if entry.variant_type != "full":
                    value = ""
            output[name] = value

        output['detail_crop_prompt_context'] = original_prompt

        for name, interpolation in self.additional_image_like_names.items():
            tensor = self._get_previous_item(variation, name, entry.base_index)
            if tensor is None:
                output[name] = None
                continue

            if entry.variant_type == 'full' or entry.source_size is None:
                output[name] = tensor
            else:
                resized_tensor = self._get_or_create_resized_tensor(
                    entry,
                    name,
                    tensor,
                    interpolation,
                )
                output[name] = self._crop_tensor(resized_tensor, entry)

        if detail_cfg['save_to_disk'] and entry.variant_type != 'full':
            self._maybe_export_tile(output[self.image_name], image_path, entry, detail_cfg, variation)

        self._decrement_remaining_cache(entry)

        return output

    # --------------------------------------------------------------------- #
    # Entry construction helpers
    # --------------------------------------------------------------------- #
    def _build_entries_for_image(
            self,
            base_image: torch.Tensor,
            detail_cfg: dict,
            base_index: int,
            base_resolution: Tuple[int, int],
    ) -> Tuple[List[DetailEntry], List[dict]]:
        entries: List[DetailEntry] = [
            DetailEntry(
                base_index=base_index,
                variant_type='full',
                scale=None,
                tile_index=0,
                top=0,
                left=0,
                tile_size=detail_cfg['tile_resolution'],
                source_size=None,
                base_resolution=base_resolution,
            )
        ]

        variant_logs: List[dict] = []

        if not detail_cfg['enabled']:
            variant_logs.append({
                'variant_type': 'detail',
                'scale': None,
                'positions': 0,
                'accepted': 0,
                'note': 'detail crops disabled in configuration',
            })
            variant_logs.append({
                'variant_type': 'context',
                'scale': 1.0,
                'positions': 0,
                'accepted': 0,
                'note': 'context tiles disabled because detail crops are disabled',
            })
            return entries, variant_logs

        tile_size = max(1, int(detail_cfg['tile_resolution']))
        overlap = max(0, int(detail_cfg['overlap']))

        base_np = self._tensor_to_numpy(base_image)

        def add_tiles_for_variant(
                variant_type: str,
                scale_value: float,
                resized_np: np.ndarray,
        ):
            variant_entries, variant_stats = self._generate_tiles_for_variant(
                base_index=base_index,
                variant_type=variant_type,
                scale_value=scale_value,
                resized_np=resized_np,
                tile_size=tile_size,
                overlap=overlap,
                detail_cfg=detail_cfg,
                base_resolution=base_resolution,
            )
            variant_logs.append(variant_stats)
            if variant_entries:
                entries.extend(variant_entries)
                key = (base_index, variant_type, scale_value)
                self._scale_remaining[key] = len(variant_entries)

        scales = self._normalize_scales(detail_cfg.get('scales', []))
        if not scales:
            variant_logs.append({
                'variant_type': 'detail',
                'scale': None,
                'positions': 0,
                'accepted': 0,
                'note': 'no scales configured',
            })
        for scale in scales:
            resized_np = self._resize_numpy(base_np, tile_size * scale)
            add_tiles_for_variant('detail', scale, resized_np)

        if detail_cfg['include_context_tiles']:
            resized_np = self._resize_numpy(base_np, tile_size)
            add_tiles_for_variant('context', 1.0, resized_np)
        else:
            variant_logs.append({
                'variant_type': 'context',
                'scale': 1.0,
                'positions': 0,
                'accepted': 0,
                'note': 'context tiles disabled in configuration',
            })

        return entries, variant_logs

    def _generate_tiles_for_variant(
            self,
            *,
            base_index: int,
            variant_type: str,
            scale_value: float,
            resized_np: np.ndarray,
            tile_size: int,
            overlap: int,
            detail_cfg: dict,
            base_resolution: Tuple[int, int],
    ) -> Tuple[List[DetailEntry], dict]:
        positions = self._enumerate_positions(resized_np.shape[1], resized_np.shape[0], tile_size, overlap)
        stats = {
            'variant_type': variant_type,
            'scale': scale_value,
            'positions': len(positions),
            'duplicates': 0,
            'blank': 0,
            'incomplete': 0,
            'accepted': 0,
        }
        if not positions:
            stats['note'] = 'no tile positions (image smaller than tile or overlap too large)'
            return [], stats

        blank_std = detail_cfg['blank_std_threshold']
        blank_edge = detail_cfg['blank_edge_threshold']
        source_size = (int(resized_np.shape[0]), int(resized_np.shape[1]))

        seen_hashes: set[bytes] = set()
        hash_lock = threading.Lock()
        stats_lock = threading.Lock() if len(positions) > 1 else None

        def _inc_stat(key: str):
            if stats_lock:
                with stats_lock:
                    stats[key] += 1
            else:
                stats[key] += 1

        def process(idx_pos: Tuple[int, Tuple[int, int]]):
            idx, (top, left) = idx_pos
            crop = resized_np[top:top + tile_size, left:left + tile_size]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                _inc_stat('incomplete')
                return None

            tile_hash = hashlib.md5(crop.tobytes()).digest()
            with hash_lock:
                if tile_hash in seen_hashes:
                    _inc_stat('duplicates')
                    return None
                seen_hashes.add(tile_hash)

            if self._is_blank_tile(crop, blank_std, blank_edge):
                _inc_stat('blank')
                return None

            entry = DetailEntry(
                base_index=base_index,
                variant_type=variant_type,
                scale=scale_value,
                tile_index=idx,
                top=top,
                left=left,
                tile_size=tile_size,
                source_size=source_size,
                base_resolution=base_resolution,
            )
            return idx, entry

        parallel_workers = int(detail_cfg.get('parallel_workers', 0) or 0)
        if parallel_workers < 0:
            parallel_workers = 0
        if parallel_workers == 0:
            parallel_workers = min(8, os.cpu_count() or 1)
        max_workers = max(1, min(parallel_workers, len(positions)))

        results: List[Tuple[int, DetailEntry]] = []
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(process, enumerate(positions)):
                    if result is not None:
                        results.append(result)
        else:
            for result in map(process, enumerate(positions)):
                if result is not None:
                    results.append(result)

        results.sort(key=lambda item: item[0])
        final_entries: List[DetailEntry] = []
        for new_index, (_, entry) in enumerate(results):
            entry.tile_index = new_index
            final_entries.append(entry)
        stats['accepted'] = len(final_entries)
        if stats['accepted'] == 0 and stats['positions'] > 0 and 'note' not in stats:
            stats['note'] = 'all tiles filtered (blank/duplicate/incomplete)'
        return final_entries, stats

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.detach().to(device='cpu', dtype=torch.float32)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        tensor = tensor.mul(255).to(dtype=torch.uint8)
        array = tensor.permute(1, 2, 0).contiguous().numpy()
        return array

    @staticmethod
    def _resize_numpy(image: np.ndarray, target_short_side: float) -> np.ndarray:
        h, w, _ = image.shape
        if h == 0 or w == 0:
            return image

        target_short_side = max(1.0, target_short_side)
        if w < h:
            new_w = int(round(target_short_side))
            new_h = max(1, int(round(h / w * target_short_side)))
        else:
            new_h = int(round(target_short_side))
            new_w = max(1, int(round(w / h * target_short_side)))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized

    @staticmethod
    def _enumerate_positions(
            width: int,
            height: int,
            tile_size: int,
            overlap: int,
    ) -> List[Tuple[int, int]]:
        step = max(1, tile_size - overlap)
        min_dim = int(tile_size * 0.3)
        positions = []
        seen = set()

        y = 0
        while y < height:
            x = 0
            while x < width:
                right = min(x + tile_size, width)
                bottom = min(y + tile_size, height)
                natural_w = right - x
                natural_h = bottom - y
                if natural_w < min_dim or natural_h < min_dim:
                    x += step
                    continue

                adj_x = x
                adj_y = y
                if natural_w < tile_size:
                    adj_x = max(0, right - tile_size)
                if natural_h < tile_size:
                    adj_y = max(0, bottom - tile_size)
                adj_x = min(adj_x, max(0, width - tile_size))
                adj_y = min(adj_y, max(0, height - tile_size))

                key = (adj_y, adj_x)
                if key not in seen:
                    seen.add(key)
                    positions.append(key)

                x += step
            y += step

        if not positions:
            positions.append((0, 0))

        return positions

    @staticmethod
    def _is_blank_tile(tile: np.ndarray, std_threshold: float, edge_threshold: int) -> bool:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        if float(np.std(gray)) < std_threshold:
            return True
        edges = cv2.Canny(gray, 100, 200)
        return int(np.count_nonzero(edges)) < edge_threshold

    @staticmethod
    def _normalize_scales(scales: Iterable) -> List[float]:
        normalized = []
        for value in scales:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number <= 0:
                continue
            normalized.append(number)
        return sorted(set(normalized))

    # --------------------------------------------------------------------- #
    # Tensor helpers
    # --------------------------------------------------------------------- #
    def _get_or_create_resized_tensor(
            self,
            entry: DetailEntry,
            tensor_name: str,
            tensor: torch.Tensor,
            interpolation: InterpolationMode,
    ) -> torch.Tensor:
        cache = self._resized_cache.setdefault(tensor_name, {})
        key = (entry.base_index, entry.variant_type, entry.scale)
        if key not in cache:
            cache[key] = self._resize_tensor(tensor, entry.source_size, interpolation)
        return cache[key]

    @staticmethod
    def _resize_tensor(
            tensor: torch.Tensor,
            target_size: Optional[Tuple[int, int]],
            interpolation: InterpolationMode,
    ) -> torch.Tensor:
        if target_size is None:
            return tensor
        h, w = target_size
        if tensor.shape[-2] == h and tensor.shape[-1] == w:
            return tensor
        antialias = interpolation in (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)
        return F.resize(
            tensor,
            [h, w],
            interpolation=interpolation,
            antialias=antialias,
        )

    @staticmethod
    def _crop_tensor(tensor: torch.Tensor, entry: DetailEntry) -> torch.Tensor:
        top = int(entry.top)
        left = int(entry.left)
        bottom = top + entry.tile_size
        right = left + entry.tile_size
        return tensor[..., top:bottom, left:right]

    def _decrement_remaining_cache(self, entry: DetailEntry):
        if entry.variant_type == 'full':
            return
        key = (entry.base_index, entry.variant_type, entry.scale)
        if key not in self._scale_remaining:
            return
        self._scale_remaining[key] -= 1
        if self._scale_remaining[key] <= 0:
            self._scale_remaining.pop(key, None)
            for name_cache in self._resized_cache.values():
                name_cache.pop(key, None)

    # --------------------------------------------------------------------- #
    # Export helpers
    # --------------------------------------------------------------------- #
    def _export_entries_to_disk(
            self,
            variation: int,
            base_image: torch.Tensor,
            image_path: str,
            entries: List[DetailEntry],
            detail_cfg: dict,
    ) -> None:
        if base_image is None:
            return
        workspace_dir = os.path.join(os.getcwd(), "workspace-cache", "detail_crops_cache")
        os.makedirs(workspace_dir, exist_ok=True)

        for entry in entries:
            if entry.variant_type == 'full':
                continue
            resized = self._get_or_create_resized_tensor(
                entry,
                self.image_name,
                base_image,
                InterpolationMode.BILINEAR,
            )
            tile_tensor = self._crop_tensor(resized, entry)
            self._write_tile_to_target(
                target_root=workspace_dir,
                variation=variation,
                tile=tile_tensor,
                original_path=image_path,
                entry=entry,
                detail_cfg=detail_cfg,
                enforce_limit=False,
            )

            if detail_cfg.get('save_to_disk'):
                user_root = detail_cfg.get('save_directory')
                if not user_root and self.export_root is not None:
                    user_root = os.path.join(self.export_root, "detail_crops")
                if user_root:
                    self._write_tile_to_target(
                        target_root=user_root,
                        variation=variation,
                        tile=tile_tensor,
                        original_path=image_path,
                        entry=entry,
                        detail_cfg=detail_cfg,
                        enforce_limit=True,
                    )

    def _maybe_export_tile(
            self,
            tile: torch.Tensor,
            original_path: str,
            entry: DetailEntry,
            detail_cfg: dict,
            variation: int,
    ):
        if not detail_cfg.get('save_to_disk'):
            return
        user_root = detail_cfg.get('save_directory')
        if not user_root and self.export_root is None:
            return
        if not user_root:
            user_root = os.path.join(self.export_root, "detail_crops")
        self._write_tile_to_target(
            target_root=user_root,
            variation=variation,
            tile=tile,
            original_path=original_path,
            entry=entry,
            detail_cfg=detail_cfg,
            enforce_limit=True,
        )

    def _write_tile_to_target(
            self,
            target_root: str,
            variation: int,
            tile: torch.Tensor,
            original_path: str,
            entry: DetailEntry,
            detail_cfg: dict,
            enforce_limit: bool,
    ) -> None:
        variation_dir = variation if detail_cfg.get('regenerate_each_epoch', False) else 0

        if enforce_limit and entry.base_index in self._export_limits:
            limit = self._export_limits[entry.base_index]
            if limit > 0 and self._export_counts[entry.base_index] >= limit:
                if entry.base_index not in self._export_limit_warnings:
                    print(
                        f"[Detail Crops] Reached save_max_tiles_per_image ({limit}) for base index {entry.base_index}; "
                        "skipping additional exports."
                    )
                    self._export_limit_warnings.add(entry.base_index)
                return

        scale_value = entry.scale if entry.scale is not None else 1.0
        scale_label = f"{float(scale_value):g}".replace('.', 'p')

        export_dir = os.path.join(target_root, "detail_crops", f"epoch-{variation_dir}", f"{entry.variant_type}_scale-{scale_label}")
        os.makedirs(export_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(original_path))[0]
        file_name = f"{base_name}_detail_{entry.variant_type}_scale-{scale_label}_tile-{entry.tile_index:03d}.png"
        export_path = os.path.join(export_dir, file_name)

        if enforce_limit and not detail_cfg.get('regenerate_each_epoch', False) and os.path.exists(export_path):
            return

        try:
            self._write_png(tile, export_path)
            if enforce_limit and entry.base_index in self._export_counts:
                self._export_counts[entry.base_index] += 1
        except Exception as exc:
            if enforce_limit:
                print(f"[Detail Crops] Failed to write crop image {export_path}: {exc}")

    @staticmethod
    def _write_png(tensor: torch.Tensor, path: str):
        tensor_cpu = tensor.detach().to(device='cpu', dtype=torch.float32)
        tensor_cpu = torch.clamp(tensor_cpu, 0.0, 1.0)
        array = tensor_cpu.mul(255).to(dtype=torch.uint8).permute(1, 2, 0).contiguous().numpy()
        cv2.imwrite(path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    # --------------------------------------------------------------------- #
    # Progress helpers
    # --------------------------------------------------------------------- #
    def _reset_progress(self, prefix: str, total: int):
        self._progress_prefix = prefix
        self._progress_total = max(0, total)
        self._progress_count = 0
        self._progress_last_percent = -1
        if self._progress_total > 0:
            self._print_progress()

    def _increment_progress(self):
        if self._progress_total <= 0:
            return
        self._progress_count = min(self._progress_count + 1, self._progress_total)
        self._print_progress()

    def _finalize_progress(self):
        if self._progress_total <= 0:
            return
        self._print_progress(final=True)

    def _print_progress(self, final: bool = False):
        if self._progress_total <= 0:
            return
        percent = int((self._progress_count * 100) / self._progress_total) if self._progress_total else 100
        if percent != self._progress_last_percent or final:
            sys.stdout.write(
                f"\r{self._progress_prefix}: {self._progress_count}/{self._progress_total} ({percent}%)"
            )
            sys.stdout.flush()
            self._progress_last_percent = percent
        if final:
            sys.stdout.write("\n")
            sys.stdout.flush()

    # --------------------------------------------------------------------- #
    # Config helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _extract_detail_config(concept: dict) -> dict:
        image_cfg = concept.get('image', {})
        detail_cfg = image_cfg.get('detail_crops', {}) or {}
        return {
            'enabled': bool(detail_cfg.get('enabled', False)),
            'tile_resolution': int(detail_cfg.get('tile_resolution', 1024) or 1024),
            'overlap': int(detail_cfg.get('overlap', 128) or 128),
            'blank_std_threshold': float(detail_cfg.get('blank_std_threshold', 10.0) or 10.0),
            'blank_edge_threshold': int(detail_cfg.get('blank_edge_threshold', 50) or 50),
            'include_context_tiles': bool(detail_cfg.get('include_context_tiles', False)),
            'regenerate_each_epoch': bool(detail_cfg.get('regenerate_each_epoch', False)),
            'enable_captioning': bool(detail_cfg.get('enable_captioning', False)),
            'caption_probability': float(detail_cfg.get('caption_probability', 0.2)),
            'caption_model': detail_cfg.get('caption_model', "qwen2.5vl:3b") or "qwen2.5vl:3b",
            'caption_system_prompt': detail_cfg.get('caption_system_prompt', "") or "",
            'caption_user_prompt': detail_cfg.get('caption_user_prompt', "") or "",
            'caption_endpoint': detail_cfg.get('caption_endpoint', "") or "",
            'caption_timeout': float(detail_cfg.get('caption_timeout', 120.0) or 120.0),
            'caption_max_retries': int(detail_cfg.get('caption_max_retries', 4) or 4),
            'caption_auto_pull': bool(detail_cfg.get('caption_auto_pull', True)),
            'scales': detail_cfg.get('scales', []) or [],
            'save_to_disk': bool(detail_cfg.get('save_to_disk', False)),
            'save_directory': detail_cfg.get('save_directory', "") or "",
            'save_max_tiles_per_image': int(detail_cfg.get('save_max_tiles_per_image', 0) or 0),
            'include_full_images': bool(detail_cfg.get('include_full_images', True)),
            'parallel_workers': int(detail_cfg.get('parallel_workers', 0) or 0),
        }

    @staticmethod
    def _build_image_identifier(original_path: str, entry: DetailEntry) -> str:
        if entry.variant_type == 'full':
            return original_path
        scale_part = "1" if entry.scale is None else f"{entry.scale:g}"
        suffix = f"#detail-{entry.variant_type}-scale{scale_part}-tile{entry.tile_index:03d}"
        return f"{original_path}{suffix}"


