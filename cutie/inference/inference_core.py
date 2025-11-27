import logging
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from cutie.inference.image_feature_store import ImageFeatureStore
from cutie.inference.memory_manager import MemoryManager
from cutie.inference.object_manager import ObjectManager
from cutie.model.cutie import CUTIE
from cutie.utils.tensor_utils import aggregate, pad_divide_by, unpad

log = logging.getLogger()


class InferenceCore:
    def __init__(
        self,
        network: CUTIE,
        cfg: DictConfig,
        *,
        image_feature_store: ImageFeatureStore = None,
    ):
        self.network = network
        self.cfg = cfg
        self.mem_every = cfg.mem_every
        stagger_updates = cfg.stagger_updates
        self.chunk_size = cfg.chunk_size
        self.save_aux = cfg.save_aux
        self.max_internal_size = cfg.max_internal_size
        self.flip_aug = cfg.flip_aug

        self.curr_ti = -1
        self.last_mem_ti = 0

        # Pre-compute stagger set once (frozenset for O(1) lookup)
        if stagger_updates >= self.mem_every:
            self.stagger_ti = frozenset(range(1, self.mem_every + 1))
        else:
            self.stagger_ti = frozenset(
                np.round(np.linspace(1, self.mem_every, stagger_updates))
                .astype(int)
                .tolist()
            )

        self.object_manager = ObjectManager()
        self.memory = MemoryManager(
            cfg=cfg, object_manager=self.object_manager
        )

        if image_feature_store is None:
            self.image_feature_store = ImageFeatureStore(self.network)
        else:
            self.image_feature_store = image_feature_store

        self.last_mask = None
        self.pad = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryManager(
            cfg=self.cfg, object_manager=self.object_manager
        )
        self.last_mask = None

    def clear_non_permanent_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_non_permanent_memory()

    def clear_sensory_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_sensory_memory()

    def update_config(self, cfg):
        self.mem_every = cfg["mem_every"]
        self.memory.update_config(cfg)

    def _add_memory(
        self,
        image: torch.Tensor,
        pix_feat: torch.Tensor,
        prob: torch.Tensor,
        key: torch.Tensor,
        shrinkage: torch.Tensor,
        selection: torch.Tensor,
        *,
        is_deep_update: bool = True,
        force_permanent: bool = False,
    ) -> None:
        """
        Memorize the given segmentation in all memory stores.

        The batch dimension is 1 if flip augmentation is not used.
        image: RGB image, (1/2)*3*H*W
        pix_feat: from the key encoder, (1/2)*_*H*W
        prob: (1/2)*num_objects*H*W, in [0, 1]
        key/shrinkage/selection: for anisotropic l2, (1/2)*_*H*W
        selection can be None if not using long-term memory
        is_deep_update: whether to use deep update (e.g. with the mask encoder)
        force_permanent: whether to force the memory to be permanent
        """
        if prob.shape[1] == 0:
            log.warn("Trying to add an empty object mask to memory!")
            return

        all_obj_ids = self.object_manager.all_obj_ids
        self.memory.initialize_sensory_if_needed(key, all_obj_ids)
        sensory = self.memory.get_sensory(all_obj_ids)

        msk_value, new_sensory, obj_value, _ = self.network.encode_mask(
            image,
            pix_feat,
            sensory,
            prob,
            deep_update=is_deep_update,
            chunk_size=self.chunk_size,
            need_weights=self.save_aux,
        )

        self.memory.add_memory(
            key,
            shrinkage,
            msk_value,
            obj_value,
            all_obj_ids,
            selection=selection,
            as_permanent="all" if force_permanent else "first",
        )
        self.last_mem_ti = self.curr_ti

        if is_deep_update:
            self.memory.update_sensory(new_sensory, all_obj_ids)

    def _segment(
        self,
        key: torch.Tensor,
        selection: torch.Tensor,
        pix_feat: torch.Tensor,
        ms_features: Iterable[torch.Tensor],
        update_sensory: bool = True,
    ) -> torch.Tensor:
        """
        Produce a segmentation using the given features and the memory

        The batch dimension is 1 if flip augmentation is not used.
        key/selection: for anisotropic l2: (1/2) * _ * H * W
        pix_feat: from the key encoder, (1/2) * _ * H * W
        ms_features: an iterable of multiscale features from the encoder, each is (1/2)*_*H*W
                      with strides 16, 8, and 4 respectively
        update_sensory: whether to update the sensory memory

        Returns: (num_objects+1)*H*W normalized probability; the first channel is the background
        """
        if not self.memory.engaged:
            log.warn("Trying to segment without any memory!")
            return torch.zeros(
                (1, key.shape[-2] * 16, key.shape[-1] * 16),
                device=key.device,
                dtype=key.dtype,
            )

        all_obj_ids = self.object_manager.all_obj_ids

        memory_readout = self.memory.read(
            pix_feat, key, selection, self.last_mask, self.network
        )
        memory_readout = self.object_manager.realize_dict(memory_readout)

        sensory, _, pred_prob_with_bg = self.network.segment(
            ms_features,
            memory_readout,
            self.memory.get_sensory(all_obj_ids),
            chunk_size=self.chunk_size,
            update_sensory=update_sensory,
        )

        # Handle flip augmentation
        if self.flip_aug:
            # In-place operations where possible
            flipped = pred_prob_with_bg[1].flip(-1)
            pred_prob_with_bg = pred_prob_with_bg[0].add_(flipped).mul_(0.5)
        else:
            pred_prob_with_bg = pred_prob_with_bg[0]

        if update_sensory:
            self.memory.update_sensory(sensory, all_obj_ids)

        return pred_prob_with_bg

    @torch.inference_mode()
    def step(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        objects: Optional[List[int]] = None,
        *,
        idx_mask: bool = True,
        end: bool = False,
        delete_buffer: bool = True,
        force_permanent: bool = False,
    ) -> torch.Tensor:
        """
        Process one frame. Returns segmentation probabilities.

        Args:
            image: 3*H*W RGB tensor
            mask: H*W (idx_mask=True) or num_objects*H*W (idx_mask=False) or None
            objects: List of object IDs corresponding to mask channels
            idx_mask: Whether mask contains object indices or separate channels
            end: If True, skip memory updates (last frame optimization)
            delete_buffer: Whether to delete cached features after this step
            force_permanent: Store this frame in permanent memory

        Returns:
            (num_objects+1)*H*W probability tensor (first channel is background)
        """
        # Infer objects from mask if not provided
        if objects is None and mask is not None:
            assert not idx_mask, (
                "Must provide objects list when using idx_mask=True"
            )
            objects = list(range(1, mask.shape[0] + 1))

        # Track original size for restoration
        orig_h, orig_w = image.shape[-2:]
        resize_needed = False

        # Resize if needed
        if self.max_internal_size > 0:
            min_side = min(orig_h, orig_w)
            if min_side > self.max_internal_size:
                resize_needed = True
                scale = self.max_internal_size / min_side
                new_h, new_w = int(orig_h * scale), int(orig_w * scale)

                image = F.interpolate(
                    image.unsqueeze(0),
                    (new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]

                if mask is not None:
                    if idx_mask:
                        mask = F.interpolate(
                            mask.float().view(1, 1, *mask.shape),
                            (new_h, new_w),
                            mode="nearest-exact",
                        )[0, 0].long()
                    else:
                        mask = F.interpolate(
                            mask.unsqueeze(0),
                            (new_h, new_w),
                            mode="bilinear",
                            align_corners=False,
                        )[0]

        self.curr_ti += 1

        # Pad to multiple of 16
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)
        if self.flip_aug:
            image = torch.cat([image, image.flip(-1)], dim=0)

        # Compute control flags once
        time_since_mem = self.curr_ti - self.last_mem_ti
        is_mem_frame = (
            (time_since_mem >= self.mem_every) or (mask is not None)
        ) and not end
        need_segment = (mask is None) or (
            self.object_manager.num_obj > 0
            and not self.object_manager.has_all(objects)
        )
        update_sensory = (time_since_mem in self.stagger_ti) and not end

        # Extract features
        ms_feat, pix_feat = self.image_feature_store.get_features(
            self.curr_ti, image
        )
        key, shrinkage, selection = self.image_feature_store.get_key(
            self.curr_ti, image
        )

        # Segment existing objects if needed
        if need_segment:
            pred_prob_with_bg = self._segment(
                key,
                selection,
                pix_feat,
                ms_feat,
                update_sensory=update_sensory,
            )

        # Process input mask
        if mask is not None:
            corresponding_tmp_ids, _ = self.object_manager.add_new_objects(
                objects
            )
            mask, _ = pad_divide_by(mask, 16)

            if need_segment:
                # Merge with existing predictions
                pred_prob_no_bg = pred_prob_with_bg[1:]

                # Clear predictions where mask is provided
                if idx_mask:
                    pred_prob_no_bg[:, mask > 0] = 0
                else:
                    pred_prob_no_bg[:, mask.max(dim=0)[0] > 0.5] = 0

                # Add new object masks
                new_masks = []
                for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                    if idx_mask:
                        this_mask = (mask == objects[mask_id]).to(
                            pred_prob_no_bg.dtype
                        )
                    else:
                        this_mask = mask[mask_id]

                    if tmp_id > pred_prob_no_bg.shape[0]:
                        new_masks.append(this_mask.unsqueeze(0))
                    else:
                        pred_prob_no_bg[tmp_id - 1] = this_mask

                if new_masks:
                    mask = torch.cat([pred_prob_no_bg, *new_masks], dim=0)
                else:
                    mask = pred_prob_no_bg

            elif idx_mask:
                # Convert index mask to one-hot
                if not objects:
                    if delete_buffer:
                        self.image_feature_store.delete(self.curr_ti)
                    log.warn("Trying to insert an empty mask as memory!")
                    return torch.zeros(
                        (1, key.shape[-2] * 16, key.shape[-1] * 16),
                        device=key.device,
                        dtype=key.dtype,
                    )
                mask = torch.stack(
                    [(mask == obj_id) for obj_id in objects], dim=0
                ).to(key.dtype)

            pred_prob_with_bg = F.softmax(aggregate(mask, dim=0), dim=0)

        # Update last_mask (detach to prevent memory leaks)
        self.last_mask = pred_prob_with_bg[1:].unsqueeze(0).detach()
        if self.flip_aug:
            self.last_mask = torch.cat(
                [self.last_mask, self.last_mask.flip(-1)], dim=0
            )

        # Add to memory
        if is_mem_frame or force_permanent:
            self._add_memory(
                image,
                pix_feat,
                self.last_mask,
                key,
                shrinkage,
                selection,
                force_permanent=force_permanent,
            )

        # Cleanup
        if delete_buffer:
            self.image_feature_store.delete(self.curr_ti)

        # Restore original size
        output_prob = unpad(pred_prob_with_bg, self.pad)
        if resize_needed:
            output_prob = F.interpolate(
                output_prob.unsqueeze(0),
                (orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0]

        return output_prob.detach()

    def delete_objects(self, objects: List[int]) -> None:
        """
        Delete the given objects from the memory.
        """
        self.object_manager.delete_objects(objects)
        self.memory.purge_except(self.object_manager.all_obj_ids)

    def output_prob_to_mask(self, output_prob: torch.Tensor) -> torch.Tensor:
        mask = torch.argmax(output_prob, dim=0)
        new_mask = torch.zeros_like(mask)
        for tmp_id, obj in self.object_manager.tmp_id_to_obj.items():
            new_mask[mask == tmp_id] = obj.id
        return new_mask
