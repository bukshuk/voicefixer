from onnxruntime import InferenceSession

import librosa
import numpy as np

from torch import cat, from_numpy

from voicefixer.tools.mel_scale import MelScale
from voicefixer.tools.pytorch_util import from_log
from voicefixer.tools.wav import save_wave


class VoiceFixer:
    def __init__(self):
        self._pre_first_stage_model = InferenceSession("models/pre01.onnx", providers=["CPUExecutionProvider"])
        self._first_stage_model = InferenceSession("models/01.onnx", providers=["CPUExecutionProvider"])
        self._second_stage_model = InferenceSession("models/02.onnx", providers=["CPUExecutionProvider"])

        self._mel_scale = MelScale(sample_rate=44100, n_stft=1025)

    def _load_wav(self, path, sample_rate):
        signal, _ = librosa.load(path, sr=sample_rate)
        return signal

    def _trim_center(self, est, ref):
        diff = np.abs(est.shape[-1] - ref.shape[-1])
        if est.shape[-1] == ref.shape[-1]:
            return est, ref
        elif est.shape[-1] > ref.shape[-1]:
            min_len = min(est.shape[-1], ref.shape[-1])
            est, ref = est[..., int(diff // 2) : -int(diff // 2)], ref
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref
        else:
            min_len = min(est.shape[-1], ref.shape[-1])
            est, ref = est, ref[..., int(diff // 2) : -int(diff // 2)]
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref

    def restore_in_memory(self, signal):
        res = []
        seg_length = 44100 * 30
        break_point = seg_length
        while break_point < signal.shape[0] + seg_length:
            segment = signal[break_point - seg_length : break_point]

            pre_first_logits = self._pre_first_stage_model.run(["output"], {"input": segment[None, None, :]})
            pre_first_out = from_numpy(pre_first_logits[0])

            mel_noisy = self._mel_scale(pre_first_out)

            first_logits = self._first_stage_model.run(["output"], {"input": mel_noisy.numpy()})
            first_out = from_numpy(first_logits[0])

            mel_enhanced = from_log(first_out)

            second_logits = self._second_stage_model.run(["output"], {"input": mel_enhanced.numpy()})
            second_out = from_numpy(second_logits[0])

            second_out, _ = self._trim_center(second_out, segment)
            res.append(second_out)
            break_point += seg_length

        second_out = cat(res, -1)

        return second_out.squeeze(0).detach().numpy()

    def restore(self, input, output):
        input_signal = self._load_wav(input, sample_rate=44100)
        output_signal = self.restore_in_memory(input_signal)
        save_wave(output_signal, fname=output, sample_rate=44100)
