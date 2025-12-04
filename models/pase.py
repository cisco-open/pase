# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import torch
import torch.nn as nn
from typing import Union, List
from .wavlm.feature_extractor import WavLM_feat as Encoder
from .vocoder.wavlmdec_dual import WavLMDec as Decoder


class PASE(nn.Module):
    def __init__(
        self, 
        wavlm_ckpt_path="/work/user_data/xiaobin/Pre-trained/PASE/DeWavLM.tar",
        wavlm_output_layer: Union[int, List[int]] = [1,24],
        vocoder_ckpt_path="/work/user_data/xiaobin/Pre-trained/PASE/Vocoder_L24.tar",
    ):
        super().__init__()
        self.encoder = Encoder(wavlm_ckpt_path, wavlm_output_layer)
        self.decoder = Decoder.from_pretrained(vocoder_ckpt_path)

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): noisy speech with shape of (B, L) or (B, 1, L)
        Return:
            y (torch.Tensor): enhanced speech with shape of (B, L).
        """
        n_samples = x.shape[-1]
        
        feat_a, feat_p = self.encoder(x)
                
        y = self.decoder(feat_p, feat_a)  # (B, 1, T)
        y = y.squeeze(1)  # (B, T)
 
        if y.shape[1] < n_samples:
            y = nn.functional.pad(y, (0, n_samples - y.shape[1]), mode='constant', value=0.0)
        else:
            y = y[..., :n_samples]
 
        return y


    
if __name__ == "__main__":

    model = PASE()

    from ptflops import get_model_complexity_info
    
    with torch.inference_mode():
        macs, params = get_model_complexity_info(model, (16000,), print_per_layer_stat=True)
    
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(macs, f"{params / 1e6:.2f} M")


