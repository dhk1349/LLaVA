import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2Processor


class Wav2VecAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'hidden')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_audio_tower', False):
            self.load_model()
        else:
            self.cfg_only = Wav2Vec2Config.from_pretrained(self.audio_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.audio_tower_name))
            return

        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.audio_tower_name)
        self.audio_tower = Wav2Vec2Model.from_pretrained(self.audio_tower_name, device_map=device_map)
        self.audio_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        if self.select_feature == 'hidden':
            # Get the hidden states from the selected layer
            audio_features = audio_forward_outs.hidden_states[self.select_layer]
        elif self.select_feature == 'all':
            # Use all hidden states
            audio_features = audio_forward_outs.hidden_states
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return audio_features

    @torch.no_grad()
    def forward(self, audio_inputs):
        if type(audio_inputs) is list:
            audio_features = []
            for audio in audio_inputs:
                audio_forward_out = self.audio_tower(
                    audio.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                audio_feature = self.feature_select(audio_forward_out).to(audio.dtype)
                audio_features.append(audio_feature)
        else:
            audio_forward_outs = self.audio_tower(
                audio_inputs.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            audio_features = self.feature_select(audio_forward_outs).to(audio_inputs.dtype)

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

