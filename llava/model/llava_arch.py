#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower, build_audio_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector, build_audio_projector

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    AUDIO_TOKEN_INDEX,
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_END_TOKEN,
)

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_vision_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

        if hasattr(config, "mm_audio_tower"):
            self.audio_tower = build_audio_tower(config, delay_load=delay_load)
            self.mm_audio_projector = build_audio_projector(config, audio_cfg=self.audio_tower.config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_tower(self):
        audio_tower = getattr(self, "audio_tower", None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, "add_faster_video"):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(
                f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
            )
            incompatible_keys = self.vision_resampler.load_state_dict(
                get_w(mm_projector_weights, "vision_resampler"), strict=False
            )
            rank0_print(
                f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
            )

    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        self.config.audio_tower = audio_tower

        if audio_tower is not None:
            from .multimodal_encoder.wav2vec_encoder import Wav2VecAudioTower

            self.audio_tower = Wav2VecAudioTower(audio_tower, model_args, delay_load=True)
            self.config.mm_audio_select_layer = model_args.mm_audio_select_layer
            self.config.mm_audio_select_feature = getattr(model_args, "mm_audio_select_feature", "hidden")

            if not hasattr(self, "audio_tower"):
                self.audio_tower = None

            if fsdp is not None and len(fsdp) > 0:
                self.audio_tower = [self.audio_tower]

            self.audio_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.audio_tower.parameters():
                p.requires_grad = True

    def initialize_audio_tokenizer(self, model_args, tokenizer):
        """Initialize tokenizer with audio-specific tokens"""
        if model_args.mm_use_audio_token:
            tokenizer.add_tokens([DEFAULT_AUDIO_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_audio_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                if model_args.tune_mm_mlp_adapter:
                    for p in self.get_input_embeddings().parameters():
                        p.requires_grad = True
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = False


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode="bilinear")

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        # Convert list to tensor if needed
        if isinstance(images, list):
            images = torch.stack(images)

        if images.ndim == 5:
            # Handle video: [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
            b, f, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            image_features = self.get_model().get_vision_tower()(images)
            # Reshape back: [batch*frames, tokens, hidden] -> [batch, frames, tokens, hidden]
            image_features = image_features.view(b, f, *image_features.shape[1:])
        else:
            image_features = self.get_model().get_vision_tower()(images)

        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_audio(self, wav):
        audio_features = self.get_model().get_audio_tower()(wav)
        audio_features = self.get_model().mm_audio_projector(audio_features)
        return audio_features

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(
            videos_or_images_features, split_sizes, dim=0
        )  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):

            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat, cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat, cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features, all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat(
            (
                image_feature,
                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device),
            ),
            dim=-1,
        )
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames, resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature = torch.cat(
            (
                image_feature,
                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device),
            ),
            dim=-1,
        )
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, audio=None, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        audio_tower = self.get_audio_tower() if hasattr(self, "get_audio_tower") else None

        if (vision_tower is None and audio_tower is None) or (images is None and audio is None) or input_ids is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Process images if present
        image_features = None
        if images is not None and vision_tower is not None:
            image_features = self.encode_images(images)

        # Process audio if present
        audio_features = None
        if audio is not None and audio_tower is not None:
            audio_features = self.encode_audio(audio)

        # Let's just add dummy tensors if they do not exist
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padding using attention_mask
        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_audio = (cur_input_ids == AUDIO_TOKEN_INDEX).sum() if audio is not None else 0

            if num_images == 0 and num_audio == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue

            # Split on both image and audio tokens
            token_indices = []
            if num_images > 0:
                token_indices.extend(
                    [(idx.item(), "image") for idx in torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]]
                )
            if num_audio > 0:
                token_indices.extend(
                    [(idx.item(), "audio") for idx in torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]]
                )
            token_indices.sort()  # Sort by position

            # Process text segments and insert features
            cur_new_input_embeds = []
            cur_new_labels = []
            last_idx = 0

            for token_idx, token_type in token_indices:
                if token_idx > last_idx:
                    # Add text segment
                    text_embeds = self.get_model().embed_tokens(cur_input_ids[last_idx:token_idx])
                    cur_new_input_embeds.append(text_embeds)
                    cur_new_labels.append(labels[batch_idx][last_idx:token_idx])

                # Add image/audio features
                if token_type == "image" and image_features is not None:
                    cur_new_input_embeds.append(image_features[batch_idx])
                    cur_new_labels.append(
                        torch.full((image_features[batch_idx].shape[0],), IGNORE_INDEX, device=labels[batch_idx].device)
                    )
                elif token_type == "audio" and audio_features is not None:
                    cur_new_input_embeds.append(audio_features[batch_idx])
                    cur_new_labels.append(
                        torch.full((audio_features[batch_idx].shape[0],), IGNORE_INDEX, device=labels[batch_idx].device)
                    )

                last_idx = token_idx + 1

            if last_idx < cur_input_ids.shape[0]:
                # Add final text segment
                text_embeds = self.get_model().embed_tokens(cur_input_ids[last_idx:])
                cur_new_input_embeds.append(text_embeds)
                cur_new_labels.append(labels[batch_idx][last_idx:])

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Handle variable length inputs
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=input_ids[0].device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_ids[0].device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if cur_len < max_len:
                # Pad embeddings
                padding = torch.zeros(
                    (max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device
                )
                new_input_embeds_padded.append(torch.cat((cur_new_embed, padding), dim=0))
            else:
                new_input_embeds_padded.append(cur_new_embed)

            # Update labels, attention mask and position ids
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels_padded

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
