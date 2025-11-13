# Copyright (c) 2024 urgent-challenge.
# Licensed under the Apache 2.0 License.
# Source: https://github.com/urgent-challenge/urgent2025_challenge
# License included under licenses/LICENSE_urgent

import ast
import re
import subprocess
from copy import deepcopy
from functools import partial
from pathlib import Path
import librosa
import numpy as np
import scipy
import soundfile as sf
import torch
from torchaudio.io import AudioEffector, CodecConfig
from tqdm.contrib.concurrent import process_map

SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
# SAMPLE_RATES = (48000, 48000)

RESAMPLE_METHODS = (
    "kaiser_best",
    "kaiser_fast",
    "scipy",
    "polyphase",
    #    "linear",
    #    "zero_order_hold",
    #    "sinc_best",
    #    "sinc_fastest",
    #    "sinc_medium",
)

OUTPUT_DIR = NotImplementedError

def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w**2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=np.bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )


def estimate_early_rir(rir_sample, early_rir_ms: float = 0.05, fs: int = 48000):
    """Estimate the part of RIR corresponding to the early reflections.

    Args:
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
        early_rir_ms (float): the duration in milliseconds that we count as early RIR
        fs (int): sampling frequency in Hz
    Returns:
        early_rir_sample (np.ndarray): estimated RIR (Channel, Time)
    """
    rir_start_sample = np.array([get_rir_start_sample(h) for h in rir_sample])
    early_rir_samples = int(early_rir_ms * fs)
    rir_stop_sample = rir_start_sample + early_rir_samples
    rir_early = rir_sample.copy()
    for i in range(rir_sample.shape[0]):
        rir_early[i, rir_stop_sample[i]:] = 0
    return rir_early


# ported from https://github.com/fgnt/sms_wsj/blob/master/sms_wsj/reverb/reverb_utils.py#L170
def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    assert level_ratio < 1, level_ratio
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        return np.min(
            [get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h]
        )

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    return rir_start_sample


ffmpeg = "/path/to/ffmpeg"


def buildFFmpegCommand(params):

    filter_commands = ""
    filter_commands += "[1:a]asplit=2[sc][mix];"
    filter_commands += (
        "[0:a][sc]sidechaincompress="
        + f"threshold={params['threshold']}:"
        + f"ratio={params['ratio']}:"
        + f"level_sc={params['sc_gain']}"
        + f":release={params['release']}"
        + f":attack={params['attack']}"
        + "[compr];"
    )
    filter_commands += "[compr][mix]amix"

    commands_list = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "quiet",
        "-i",
        params["speech_path"],
        "-i",
        params["noise_path"],
        "-filter_complex",
        filter_commands,
        "-strict", "normal",
        params["output_path"],
    ]

    return commands_list


#############################
# Augmentations per sample
#############################
def mix_noise(speech_sample, noise_sample, snr=5.0, rng=None):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (Channel, Time)
        noise_sample (np.ndarray): a single noise sample (Channel, Time)
        snr (float): signal-to-nosie ratio (SNR) in dB
        rng (np.random.Generator): random number generator
    Returns:
        noisy_sample (np.ndarray): output noisy sample (Channel, Time)
        noise (np.ndarray): scaled noise sample (Channel, Time)
    """
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        # Repeat noise
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]

    power_speech = (speech_sample[detect_non_silence(speech_sample)] ** 2).mean()
    power_noise = (noise_sample[detect_non_silence(noise_sample)] ** 2).mean()
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample
    noisy_speech = speech_sample + noise
    return noisy_speech, noise


def wind_noise(
    speech_sample,
    noise_sample,
    fs,
    uid,
    threshold,
    ratio,
    attack,
    release,
    sc_gain,
    clipping,
    clipping_threshold,
    snr,
    rng=None,
):
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        # Repeat noise
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]

    power_speech = (speech_sample[detect_non_silence(speech_sample)] ** 2).mean()
    power_noise = (noise_sample[detect_non_silence(noise_sample)] ** 2).mean()
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample

    # to use ffmpeg for simulation, speech and noise have to be saved once
    tmp_dir = Path(f"{OUTPUT_DIR}/simulation_tmp")
    tmp_dir.mkdir(exist_ok=True)
    speech_tmp_path = tmp_dir / f"speech_{uid}.wav"
    noise_tmp_path = tmp_dir / f"noise_{uid}.wav"
    mix_tmp_path = tmp_dir / f"mix_{uid}.wav"

    scale = 0.9 / max(
        np.max(np.abs(speech_sample)),
        np.max(np.abs(noise)),
    )
    speech_sample *= scale
    noise *= scale

    save_audio(speech_sample, speech_tmp_path, fs)
    save_audio(noise, noise_tmp_path, fs)

    commands = buildFFmpegCommand(
        {
            "speech_path": speech_tmp_path,
            "noise_path": noise_tmp_path,
            "output_path": mix_tmp_path,
            "threshold": threshold,
            "ratio": ratio,
            "attack": attack,
            "release": release,
            "sc_gain": sc_gain,
        }
    )

    if subprocess.run(commands).returncode != 0:
        print("There was an error running your FFmpeg script")

    # Clipper
    mix, sr = sf.read(mix_tmp_path)
    noise, sr = sf.read(noise_tmp_path)

    mix /= scale
    noise /= scale

    if clipping:
        mix = np.maximum(clipping_threshold * np.min(mix) * np.ones_like(mix), mix)
        mix = np.minimum(clipping_threshold * np.max(mix) * np.ones_like(mix), mix)

    return mix[None], noise[None]


def add_reverberation(speech_sample, rir_sample):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
    Returns:
        reverberant_sample (np.ndarray): output noisy sample (Channel, Time)
    """
    reverberant_sample = scipy.signal.convolve(speech_sample, rir_sample, mode="full")
    return reverberant_sample[:, : speech_sample.shape[1]]


def gen_bandwidth_limitation_params(fs: int = 16000, res_type="random"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        fs (int): sampling rate in Hz
        res_type (str): resampling method

    Returns:
        res_type (str): adopted resampling method
        fs_new (int): effective sampling rate in Hz
    """
    # resample to a random sampling rate
    fs_opts = [fs_new for fs_new in SAMPLE_RATES if fs_new < fs]
    if fs_opts:
        if res_type == "random":
            res_type = np.random.choice(RESAMPLE_METHODS)
        fs_new = np.random.choice(fs_opts)
        opts = {"res_type": res_type}
    else:
        res_type = "none"
        fs_new = fs
    return res_type, fs_new


def bandwidth_limitation(speech_sample, fs: int, fs_new: int, res_type="kaiser_best"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        fs (int): sampling rate in Hz
        fs_new (int): effective sampling rate in Hz
        res_type (str): resampling method

    Returns:
        ret (np.ndarray): bandwidth-limited speech sample (1, Time)
    """
    opts = {"res_type": res_type}
    if fs == fs_new:
        return speech_sample
    assert fs > fs_new, (fs, fs_new)
    ret = librosa.resample(speech_sample, orig_sr=fs, target_sr=fs_new, **opts)
    # resample back to the original sampling rate
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return ret[:, : speech_sample.shape[1]]


def clipping(speech_sample, min_quantile: float = 0.0, max_quantile: float = 0.9):
    """Apply the clipping distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        min_quantile (float): lower bound on the quantile of samples to be clipped
        max_quantile (float): upper bound on the quantile of samples to be clipped

    Returns:
        ret (np.ndarray): clipped speech sample (1, Time)
    """
    q = np.array([min_quantile, max_quantile])
    min_, max_ = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    # per-channel clipping
    ret = np.stack(
        [
            np.clip(speech_sample[i], min_[i], max_[i])
            for i in range(speech_sample.shape[0])
        ],
        axis=0,
    )
    return ret


"""
def codec_compression(speech_sample, fs: int, vbr_quality: float):
    # if random.random() > 0.5:
    #     module = Pedalboard([GSMFullRateCompressor()])
    # else:
    #     module = Pedalboard([MP3Compressor()])
    # vbr_quality = random.uniform(
    #     params["mp3_vbr_quality"][0], params["mp3_vbr_quality"][1]
    # )
    # print(vbr_quality)
    assert 0.0 <= vbr_quality <= 10.0
    module = Pedalboard([MP3Compressor(vbr_quality=vbr_quality)])
    output = module(speech_sample, fs)
    return output
"""


def codec_compression(
    speech_sample,
    fs: int,
    format: str,
    encoder: str = None,
    qscale: int = None,
):
    assert format in ["mp3", "ogg"], format
    assert encoder in [None, "None", "vorbis", "opus"], encoder

    encoder = None if encoder == "None" else encoder
    if speech_sample.ndim == 2:
        speech_sample = speech_sample.T  # (channel, sample) -> (sample, channel)
    try:
        module = AudioEffector(
            format=format,
            encoder=encoder,
            codec_config=CodecConfig(qscale=qscale),
            pad_end=True,
        )
        output = module.apply(torch.tensor(speech_sample), fs).numpy()
    except Exception as e:
        print(format, encoder, qscale, flush=True)
        print(e, flush=True)

    if output.shape[0] < speech_sample.shape[0]:
        zeros = np.zeros((speech_sample.shape[0] - output.shape[0], output.shape[1]))
        output = np.concatenate((output, zeros), axis=0)
    elif output.shape[0] > speech_sample.shape[0]:
        output = output[: speech_sample.shape[0]]

    assert speech_sample.shape == output.shape, (speech_sample.shape, output.shape)
    return (
        output.T if output.ndim == 2 else output
    )  # (sample, channel) -> (channel, sample)



def gen_packet_loss_params(
    speech_length, fs, packet_duration_ms, packet_loss_rate, max_continuous_packet_loss, rng
):
    """Returns a list of indices (of packets) that are zeroed out."""

    # speech duration in ms and the number of packets
    speech_duration_ms = speech_length / fs * 1000
    num_packets = int(speech_duration_ms // packet_duration_ms)

    # randomly select the packet loss rate and calculate the packet loss duration
    packet_loss_rate = rng.uniform(*packet_loss_rate)
    packet_loss_duration_ms = packet_loss_rate * speech_duration_ms

    # calculate the number of packets to be zeroed out
    num_packet_loss = int(round(packet_loss_duration_ms / packet_duration_ms, 0))

    # list of length of each packet loss
    packet_loss_lengths = []
    for _ in range(num_packet_loss):
        num_continuous_packet_loss = rng.integers(1, max_continuous_packet_loss, endpoint=True)
        packet_loss_lengths.append(num_continuous_packet_loss)

        if num_packet_loss - sum(packet_loss_lengths) <= max_continuous_packet_loss:
            packet_loss_lengths.append(num_packet_loss - sum(packet_loss_lengths))
            break

    packet_loss_start_indices = rng.choice(
        range(num_packets), len(packet_loss_lengths), replace=False
    )
    packet_loss_indices = []
    for idx, length in zip(packet_loss_start_indices, packet_loss_lengths):
        packet_loss_indices += list(range(idx, idx + length))

    return list(set(packet_loss_indices))


def packet_loss(
    speech_sample, fs: int, packet_loss_indices: list, packet_duration_ms: int = 20
):
    for idx in packet_loss_indices:
        start = idx * packet_duration_ms * fs // 1000
        end = (idx + 1) * packet_duration_ms * fs // 1000
        speech_sample[:, start:end] = 0

    return speech_sample



def get_pl_target(noisy_speech, speech_sample, delta_SNR=5.0, K=4):
    """Generate PL target.

    This function generates the PL target based on the given noisy speech and clean speech samples.
    The resulting `noisy_speech` has a shape of [K+2, wav_len], where `K` is the number of intermediate targets in PL.
    Each row of `noisy_speech` corresponds to the noisy signal, signals with increasing SNR by delta_SNR, and the clean speech signal, respectively.

    Args:
        noisy_speech (ndarray): The noisy speech signal.
        speech_sample (ndarray): The clean speech signal.
        delta_SNR (float, optional): The increment in SNR (Signal-to-Noise Ratio) for each intermediate target. Defaults to 5.0.
        K (int, optional): The number of intermediate targets in PL. Defaults to 4.
    Returns:
        ndarray: The PL target with shape [K+2, wav_len].
    """
    noise_sample = noisy_speech - speech_sample
    snr_values = np.arange(0, (K+1) * delta_SNR, delta_SNR)
    scales = np.concatenate((10 ** (-snr_values / 20), [0]))
    noise_sample = scales[:, np.newaxis] * noise_sample
    noisy_speech = speech_sample + noise_sample
    return noisy_speech


#############################
# Audio utilities
#############################
def read_audio(filename, force_1ch=False, fs=None, start=0, stop=None):
    audio, fs_ = sf.read(filename, start=start, stop=stop, always_2d=True)
    audio = audio[:, :1].T if force_1ch else audio.T
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
        return audio, fs
    return audio, fs_


def save_audio(audio, filename, fs):
    if audio.ndim != 1:
        audio = audio[0] if audio.shape[0] == 1 else audio.T
    sf.write(filename, audio, samplerate=fs)


    

