# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import torch
from torch import nn
from omegaconf import OmegaConf
from .vocos.backbone import VocosBackbone as Decoder
from .vocos.head import ISTFTHead as Head


class WavLMDec(nn.Module):
    def __init__(
        self,
        input_channels=1024,
        dim=768,
        intermediate_dim=2304,
        num_layers=12,
        n_fft=1280,
        hop_length=320
    ):
        super().__init__()
        self.decoder = Decoder(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers
        )
        self.head = Head(
            dim=dim,
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    @classmethod
    def from_pretrained(cls, ckpt_path: str, frozen: bool = True):
        print("[Vocoder] Loading from:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        model = cls(**ckpt['cfg'])
        model.load_state_dict(ckpt['model'], strict=False)
        
        if frozen:
            model = model.eval()
            for p in model.parameters():
                p.requires_grad = False
        
        return model
        

    def forward(self, embed):
        """embed: (B, T, D)"""
        x = embed.transpose(1, 2)  # (B, D, T)
        x = self.decoder(x)
        audio_output = self.head(x)
    
        return audio_output
    
    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train_vocoder.yaml')

    wavlmdec = WavLMDec(**config['decoder_config'])
    
    x = torch.randn(1, 1024, 50)
    out = wavlmdec(x)
    print(out.shape)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(wavlmdec, (1024, 50), as_strings=True,
                                            print_per_layer_stat=False, verbose=False)
    print(f"MACs: {macs}, Params: {params}")