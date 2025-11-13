# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import torch
import torch.nn as nn
from typing import Union, List
from .WavLM import WavLMConfig, WavLM


class WavLM_feat(nn.Module):
    def __init__(
        self,
        wavlm_ckpt_path="/work/user_data/xiaobin/Pre-trained/WavLM/WavLM-Large.pt",
        output_layer: Union[int, List[int]] = 24,
        load_pretrained: bool=True,
        freeze: bool=True,
    ):
        super().__init__()
        self.wavlm_ckpt_path = wavlm_ckpt_path
        cpt = torch.load(self.wavlm_ckpt_path, map_location="cpu")
        print("WavLM output layer:", output_layer)
        
        self.cfg = WavLMConfig(cpt['cfg'])
        self.wavlm = WavLM(self.cfg)
        if load_pretrained:
            self.wavlm.load_state_dict(cpt['model'])
            print("Loading WavLM from:", wavlm_ckpt_path)

        if freeze:
            self.wavlm.eval()
            for p in self.wavlm.parameters():
                p.requires_grad = False

        self.output_layer = output_layer

    @staticmethod
    def pad(x):
        if x.shape[1] % 320 != 80:
            pad_points = x.shape[1]//320 * 320 + 80 - x.shape[1]
            x = nn.functional.pad(x, [0, pad_points])
        return x

    def forward(self, wav, mask=False, mask_indices=None):
        """wav: (B, 1, L)"""
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        
        wav = self.pad(wav)
        
        L = self.output_layer if isinstance(self.output_layer, int) else max(self.output_layer)
        
        res = self.wavlm.extract_features(wav, output_layer=L, mask=mask, mask_indices=mask_indices)[0]
        layer_reps = res["layer_reps"]
        
        if isinstance(self.output_layer, int) or len(self.output_layer) == 1:
            feat = layer_reps[L]
            feat = torch.nn.functional.layer_norm(feat, feat.shape, eps=1e-6)
        else:
            feat = []
            for i in range(len(self.output_layer)):
                feat_i = layer_reps[self.output_layer[i]]
                feat_i = torch.nn.functional.layer_norm(feat_i, feat_i.shape, eps=1e-6)
                feat.append(feat_i)

        return feat
    

if __name__ == "__main__":
    feature_extractor = WavLM_feat(output_layer=[1, 24])
   
    # print(dir(feature_extractor.cfg)) 
    
    for k in dir(feature_extractor.cfg):
        # if 'mask' in k:
        print(k, getattr(feature_extractor.cfg, k))
    # x = torch.randn(1, 16000)
    # y = feature_extractor(x)
    # print(y.shape)