from onnxruntime import InferenceSession

import librosa
import numpy as np

from torch import from_numpy, cat, clamp, pow, Tensor

from voicefixer.tools.wav import save_wave


SAMPLE_RATE = 44100


class VoiceFixer:
    def __init__(self):
        self._pre_first_stage_model = InferenceSession("models/pre_01.onnx", providers=["CPUExecutionProvider"])
        self._pre_second_stage_model = InferenceSession("models/pre_02.onnx", providers=["CPUExecutionProvider"])
        self._first_stage_model = InferenceSession("models/01.onnx", providers=["CPUExecutionProvider"])
        self._second_stage_model = InferenceSession("models/02.onnx", providers=["CPUExecutionProvider"])

    def restore(self, input_path, output_path):
        input_signal, _ = librosa.load(input_path, sr=SAMPLE_RATE)

        output_signal = self.restore_in_memory(input_signal)

        save_wave(output_signal, output_path, SAMPLE_RATE)

    def restore_in_memory(self, signal: np.ndarray):
        res = []
        seg_length = SAMPLE_RATE * 30
        break_point = seg_length
        while break_point < signal.shape[0] + seg_length:
            segment = signal[break_point - seg_length : break_point]

            pre_first_out = self.run_onnx_model(self._pre_first_stage_model, from_numpy(segment.reshape(1, 1, -1)))

            pre_second_out = self.run_onnx_model(self._pre_second_stage_model, pre_first_out)

            first_out = self.run_onnx_model(self._first_stage_model, pre_second_out)
            first_out = pow(10, clamp(first_out, min=-np.inf, max=5))

            second_out = self.run_onnx_model(self._second_stage_model, first_out)
            second_out, _ = self.trim_center(second_out, segment)

            res.append(second_out)
            break_point += seg_length

        second_out = cat(res, -1)

        return second_out.squeeze(0).detach().numpy()

    def run_onnx_model(self, model: InferenceSession, input: Tensor) -> Tensor:
        return from_numpy(model.run(["output"], {"input": input.numpy()})[0])

    def trim_center(self, est, ref):
        est_len = est.shape[-1]
        ref_len = ref.shape[-1]
        min_len = min(est_len, ref_len)
        pad = int(np.abs(est_len - ref_len) // 2)

        if est_len > min_len:
            est = est[..., pad:]
            est = est[..., :min_len]

        if ref_len > min_len:
            ref = ref[..., pad:]
            ref = ref[..., :min_len]

        return est, ref
