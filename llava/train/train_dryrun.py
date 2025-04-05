import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers

from llava.model import LlavaLlamaForCausalLM
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    AUDIO_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_END_TOKEN,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14")
    audio_tower: Optional[str] = field(default="facebook/wav2vec2-base")
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_audio_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_use_im_start_end: bool = field(default=True)
    mm_use_audio_start_end: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: str = field(default="spatial")
    add_faster_video: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    image_folder: Optional[str] = field(default="dummy_images")
    audio_folder: Optional[str] = field(default="dummy_audio")
    is_multimodal: bool = field(default=True)
    image_aspect_ratio: str = field(default="square")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=512)
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)


# Simplified mock dataset that includes video frames and audio tokens
class MockDataset(Dataset):
    def __init__(self, tokenizer, model_args, data_args, size=10, num_frames=4):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.size = size
        self.num_frames = num_frames

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Create dummy video frames (batch of frames)
        video = torch.randn(self.num_frames, 3, 224, 224)  # [num_frames, channels, height, width]
        audio = torch.randn(16000)  # Mock 1 second audio

        # Create conversation with video and audio tokens
        image_token = DEFAULT_IMAGE_TOKEN
        audio_token = DEFAULT_AUDIO_TOKEN

        if self.model_args.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
        if self.model_args.mm_use_audio_start_end:
            audio_token = DEFAULT_AUDIO_START_TOKEN + audio_token + DEFAULT_AUDIO_END_TOKEN

        # Simple conversation format with video context
        conversation = [
            {"from": "human", "value": f"Look at this video {image_token} and listen to this {audio_token}"},
            {"from": "assistant", "value": "I see the video frames and hear the audio. This is a test response."},
        ]

        # Tokenize
        conv_text = conversation[0]["value"] + conversation[1]["value"]
        tokenized = self.tokenizer(
            conv_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
        )

        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        labels = input_ids.clone()

        # Mask labels for human parts
        human_text_len = len(self.tokenizer(conversation[0]["value"]).input_ids)
        labels[:human_text_len] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image": video,  # Now returning video frames instead of single image
            "audio": audio,
        }


def train_dryrun():
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set video-specific configurations
    model_args.mm_patch_merge_type = "spatial"  # Required for video processing
    model_args.add_faster_video = True  # Enable video processing
    data_args.is_multimodal = True

    # Initialize tokenizer with minimal settings
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Add special tokens
    special_tokens = {
        "additional_special_tokens": [
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_AUDIO_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_AUDIO_START_TOKEN,
            DEFAULT_AUDIO_END_TOKEN,
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")

    # Initialize LLaVA model with minimal settings
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float32,  # Use float32 for CPU
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize vision modules with minimal settings
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)  # No FSDP for simple testing

        vision_tower = model.get_vision_tower()
        data_args.image_processor = vision_tower.image_processor

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_patch_merge_type = model_args.mm_patch_merge_type
        model.config.add_faster_video = model_args.add_faster_video
        model.config.mm_spatial_pool_stride = 2  # Add this for video pooling

        # Initialize vision tokenizer
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Create dataset and dataloader with video frames
    train_dataset = MockDataset(tokenizer=tokenizer, model_args=model_args, data_args=data_args, size=10, num_frames=4)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    print("Starting dry run training with video data...")
    for epoch in range(2):  # Just 2 epochs for testing
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Process video frames
            video_frames = batch["image"]  # Shape: [batch_size, num_frames, channels, height, width]

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                images=[frames for frames in video_frames],  # Pass as list of frame tensors
                audio=batch["audio"],
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            if batch_idx >= 10:  # Only run few batches for testing
                break

    print("Video dry run completed successfully!")


if __name__ == "__main__":
    train_dryrun()
