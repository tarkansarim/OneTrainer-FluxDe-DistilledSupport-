import json
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState

import torch


class DataLoaderMgdsMixin(metaclass=ABCMeta):

    def _create_mgds(
            self,
            config: TrainConfig,
            definition: list,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        concepts = config.concepts
        if concepts is None:
            with open(config.concept_file_name, 'r') as f:
                raw_concepts = json.load(f)
                print(f"[DataLoader] Loading concepts from file: {config.concept_file_name}, found {len(raw_concepts)} raw concepts")
                concepts = [ConceptConfig.default_values().from_dict(c) for c in raw_concepts]
        else:
            print(f"[DataLoader] Using concepts from config: {len(concepts)} concepts")

        print(f"[DataLoader] Before validation filter: {len(concepts)} concepts")
        for i, concept in enumerate(concepts):
            concept_type = getattr(concept, 'type', 'unknown')
            print(f"[DataLoader] Concept {i}: type={concept_type}, name={getattr(concept, 'name', 'unnamed')}")
            # Augmentation debug summary (only when debug_mode is enabled)
            if getattr(config, 'debug_mode', False):
                try:
                    img = getattr(concept, 'image', None)
                    txt = getattr(concept, 'text', None)
                    if img is not None:
                        print(
                            "[Augmentations] Concept {idx} image: "
                            "crop_jitter={crop_jitter}, "
                            "flip(random={rf}, fixed={ff}), "
                            "rotate(random={rr}, fixed={fr}, max_angle={rmax}), "
                            "brightness(random={rb}, fixed={fb}, max={rbmax}), "
                            "contrast(random={rc}, fixed={fc}, max={rcmax}), "
                            "saturation(random={rs}, fixed={fs}, max={rsmax}), "
                            "hue(random={rh}, fixed={fh}, max={rhmax}), "
                            "mask_shrink={ms}, mask_rotate_crop={mrc}"
                            .format(
                                idx=i,
                                crop_jitter=getattr(img, 'enable_crop_jitter', False),
                                rf=getattr(img, 'enable_random_flip', False),
                                ff=getattr(img, 'enable_fixed_flip', False),
                                rr=getattr(img, 'enable_random_rotate', False),
                                fr=getattr(img, 'enable_fixed_rotate', False),
                                rmax=getattr(img, 'random_rotate_max_angle', 0.0),
                                rb=getattr(img, 'enable_random_brightness', False),
                                fb=getattr(img, 'enable_fixed_brightness', False),
                                rbmax=getattr(img, 'random_brightness_max_strength', 0.0),
                                rc=getattr(img, 'enable_random_contrast', False),
                                fc=getattr(img, 'enable_fixed_contrast', False),
                                rcmax=getattr(img, 'random_contrast_max_strength', 0.0),
                                rs=getattr(img, 'enable_random_saturation', False),
                                fs=getattr(img, 'enable_fixed_saturation', False),
                                rsmax=getattr(img, 'random_saturation_max_strength', 0.0),
                                rh=getattr(img, 'enable_random_hue', False),
                                fh=getattr(img, 'enable_fixed_hue', False),
                                rhmax=getattr(img, 'random_hue_max_strength', 0.0),
                                ms=getattr(img, 'enable_random_circular_mask_shrink', False),
                                mrc=getattr(img, 'enable_random_mask_rotate_crop', False),
                            )
                        )
                    if txt is not None:
                        print(
                            "[Augmentations] Concept {idx} text: "
                            "tag_shuffle={ts}, "
                            "tag_dropout(enable={td}, prob={tdp}, mode={tdm}, special_mode={tdsm}, regex={tdsr}), "
                            "caps_randomize(enable={cr}, prob={crp}, mode={crm}, lowercase={crl})"
                            .format(
                                idx=i,
                                ts=getattr(txt, 'enable_tag_shuffling', False),
                                td=getattr(txt, 'tag_dropout_enable', False),
                                tdp=getattr(txt, 'tag_dropout_probability', 0.0),
                                tdm=getattr(txt, 'tag_dropout_mode', ''),
                                tdsm=getattr(txt, 'tag_dropout_special_tags_mode', ''),
                                tdsr=getattr(txt, 'tag_dropout_special_tags_regex', False),
                                cr=getattr(txt, 'caps_randomize_enable', False),
                                crp=getattr(txt, 'caps_randomize_probability', 0.0),
                                crm=getattr(txt, 'caps_randomize_mode', ''),
                                crl=getattr(txt, 'caps_randomize_lowercase', False),
                            )
                        )
                except Exception as e:
                    print(f"[Augmentations] Warning: failed to print augmentation summary for concept {i}: {e}")

        # choose all validation concepts, or none of them, depending on is_validation
        concepts = [concept for concept in concepts if (ConceptType(concept.type) == ConceptType.VALIDATION) == is_validation]

        print(f"[DataLoader] After validation filter (is_validation={is_validation}): {len(concepts)} concepts")

        # convert before passing to MGDS
        concepts = [c.to_dict() for c in concepts]

        settings = {
            "target_resolution": config.resolution,
            "target_frames": config.frames,
        }

        # Just defaults for now.
        # Use self.train_device instead of config.train_device to respect device_indexes
        ds = MGDS(
            self.train_device,
            concepts,
            settings,
            definition,
            batch_size=config.batch_size, #local batch size
            state=PipelineState(config.dataloader_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
