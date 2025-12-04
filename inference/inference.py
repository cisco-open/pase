# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import torch
import os
import soundfile as sf
from tqdm import tqdm

from models.pase import PASE

wavlm_ckpt_path = '/work/user_data/xiaobin/Pre-trained/PASE/DeWavLM.tar'
vocoder_ckpt_path = '/work/user_data/xiaobin/Pre-trained/PASE/Vocoder_dual.tar'


def inference_file(input_file, output_file, model):
    """
    Run inference on a single audio file and save the result.
    Args:
        input_file (str): Path to input audio file.
        output_file (str): Path to save enhanced audio.
        model: Initialized model for inference.
    """
    audio, fs = sf.read(input_file, dtype='float32')
    input_tensor = torch.FloatTensor(audio).unsqueeze(
        0).to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model(input_tensor)
    enhanced = output.cpu().detach().numpy().squeeze()
    sf.write(output_file, enhanced, fs)


def inference_folder(input_dir, output_dir, model, extension='.wav'):
    """
    Run inference on all audio files in a folder and save results to output_dir.
    Args:
        input_dir (str): Directory with input audio files.
        output_dir (str): Directory to save enhanced files.
        model: Initialized model for inference.
        extension (str): File extension to filter (default: '.wav').
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in tqdm(os.listdir(input_dir)):
        if fname.lower().endswith(extension):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            inference_file(in_path, out_path, model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run PASE inference on audio files.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for enhanced files')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('--extension', type=str, default='.wav',
                        help='Audio file extension (default: .wav)')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PASE(
        wavlm_ckpt_path=wavlm_ckpt_path,
        wavlm_output_layer=[1, 24],
        vocoder_ckpt_path=vocoder_ckpt_path,
    ).to(device).eval()

    inference_folder(args.input_dir, args.output_dir,
                     model, extension=args.extension)
