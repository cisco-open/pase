# PASE: Phonologically Anchored Speech Enhancer
PASE presents a state-of-the-art generative model for speech enhancement, with its notably low hallucination being a key advantage.

[![arxiv]()]()
[![demo]()]()
[![models]()]()


## Inference
To run inference on audio files, use:

```bash
python -m inference.inference --input_dir <input_dir> --output_dir <output_dir> [options]
```

| Argument       | Requirement / Default | Description                                                  |
|----------------|-----------------------|--------------------------------------------------------------|
| `--input_dir`  | **required**          | Path to the input directory containing audio files.          |
| `--output_dir` | **required**          | Path to the output directory where enhanced files will be saved. |
| `--device`     | default: `cuda:0`     | Torch device to run inference on, e.g., `cuda:0`, `cuda:1`, or `cpu`. |
| `--extension`  | default: `.wav`       | Audio file extension to process.                             |

## Training
### Step 1: Training a single-stream vocoder
- training script: `train/train_vocoder.py`
- training configuration: `configs/cfg_train_vocoder.yaml`

    ```bash
    python -m train.train_vocoder -C configs/cfg_train_vocoder.yaml -D 0,1,2,3
    ```
- inference script: `inference/infer_vocoder.py`
    ```bash
    python -m inference.infer_vocoder -C configs/cfg_infer.yaml -D 0
    ```

This step aims to pre-train a vocoder using the 24th-layer WavLM representations. The pre-trained single-stream vocoder is then used in Step 2 to reconstruct waveforms, enabling the evaluation of DeWavLM’s performance.

### Step 2: Fine-tuning WavLM
- training script: `train/train_wavlm.py`
- training configuration: `configs/cfg_train_wavlm.yaml`
- inference script: `inference/infer_wavlm.py`

(The usage is the same as in Step 1.)

This step aims to obtain a denoised WavLM (DeWavLM) via knowledge distillation, referred to in the paper as denoising representation distillation (DRD).

### Step 3: Training a dual-stream vocoder
- training script: `train/train_vocoder_dual.py`
- training configuration: `configs/cfg_train_vocoder_dual.yaml`
- inference script: `inference/infer_vocoder_dual.py`

(The usage is the same as in Step 1.)

This step trains the final dual-stream vocoder, which takes the acoustic (1st-layer) and phonetic (24th-layer) DeWavLM representations as inputs and produces the final enhanced waveform.

## Citation
If you find this work useful, please cite our paper:
```bibtex

```