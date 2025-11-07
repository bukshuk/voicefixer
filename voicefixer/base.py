from onnxruntime import InferenceSession

from voicefixer.tools.mel_scale import MelScale
from voicefixer.tools.modules.fDomainHelper import FDomainHelper
from voicefixer.tools.pytorch_util import *
from voicefixer.tools.wav import *

import json


class VoiceFixer:
    def __init__(self):
        self._first_stage_model = InferenceSession("models/01.onnx", providers=["CPUExecutionProvider"])
        self._second_stage_model = InferenceSession("models/02.onnx", providers=["CPUExecutionProvider"])

        self._f_domain_helper = FDomainHelper()
        self._mel_scale = MelScale(sample_rate=44100, n_stft=1025)

    def _load_wav_energy(self, path, sample_rate, threshold=0.95):
        wav_10k, _ = librosa.load(path, sr=sample_rate)
        stft = np.log10(np.abs(librosa.stft(wav_10k)) + 1.0)
        fbins = stft.shape[0]
        e_stft = np.sum(stft, axis=1)
        for i in range(e_stft.shape[0]):
            e_stft[-i - 1] = np.sum(e_stft[: -i - 1])
        total = e_stft[-1]
        for i in range(e_stft.shape[0]):
            if e_stft[i] < total * threshold:
                continue
            else:
                break
        return wav_10k, int((sample_rate // 2) * (i / fbins))

    def _load_wav(self, path, sample_rate, threshold=0.95):
        wav_10k, _ = librosa.load(path, sr=sample_rate)
        return wav_10k

    def _amp_to_original_f(self, mel_sp_est, mel_sp_target, cutoff=0.2):
        freq_dim = mel_sp_target.size()[-1]
        mel_sp_est_low, mel_sp_target_low = (
            mel_sp_est[..., 5 : int(freq_dim * cutoff)],
            mel_sp_target[..., 5 : int(freq_dim * cutoff)],
        )
        energy_est, energy_target = torch.mean(mel_sp_est_low, dim=(2, 3)), torch.mean(mel_sp_target_low, dim=(2, 3))
        amp_ratio = energy_target / energy_est
        return mel_sp_est * amp_ratio[..., None, None], mel_sp_target

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

    def _pre(self, input, cuda):
        input = input[None, None, ...]
        input = torch.tensor(input)
        input = try_tensor_cuda(input, cuda=cuda)
        sp, _, _ = self._f_domain_helper.wav_to_spectrogram_phase(input)
        mel_orig = self._mel_scale(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return mel_orig

    @torch.no_grad()
    def restore_in_memory(self, wav_10k, cuda=False):
        res = []
        seg_length = 44100 * 30
        break_point = seg_length
        while break_point < wav_10k.shape[0] + seg_length:
            segment = wav_10k[break_point - seg_length : break_point]

            mel_noisy = self._pre(segment, cuda)

            # self.save_to_json(mel_noisy, "01_in")
            first_logits = self._first_stage_model.run(["output"], {"input": mel_noisy.numpy()})
            first_out = torch.from_numpy(first_logits[0])
            # self.save_to_json(first_out, "01_out")

            mel_enhanced = from_log(first_out)

            # self.save_to_json(mel_enhanced, "02_in")
            second_logits = self._second_stage_model.run(["output"], {"input": mel_enhanced.numpy()})
            second_out = torch.from_numpy(second_logits[0])
            # self.save_to_json(second_out, "02_out")

            # unify energy
            if torch.max(torch.abs(second_out)) > 1.0:
                second_out = second_out / torch.max(torch.abs(second_out))
                print("Warning: Exceed energy limit,", input)
            # frame alignment
            second_out, _ = self._trim_center(second_out, segment)
            res.append(second_out)
            break_point += seg_length
        second_out = torch.cat(res, -1)

        return tensor2numpy(second_out.squeeze(0))

    def restore(self, input, output, cuda=False, mode=0, your_vocoder_func=None):
        wav_10k = self._load_wav(input, sample_rate=44100)
        out_np_wav = self.restore_in_memory(wav_10k, cuda=cuda)
        save_wave(out_np_wav, fname=output, sample_rate=44100)

    def save_to_json(self, data, name):
        try:
            with open(f"{name}.json", "w") as f:
                json.dump(data.numpy().flatten().tolist(), f, indent=4)
                print(f"Tensor {data.shape} successfully saved to {name}.json")
        except IOError as e:
            print(f"Error writing to file: {e}")
