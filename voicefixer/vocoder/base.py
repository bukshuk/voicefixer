from onnxruntime import InferenceSession

from torch import from_numpy

from voicefixer.vocoder.model.generator import Generator
from voicefixer.tools.pytorch_util import *
from voicefixer.vocoder.model.util import *
from voicefixer.vocoder.config import Config


class Vocoder(nn.Module):
    def __init__(self, sample_rate):
        super(Vocoder, self).__init__()

        self.rate = sample_rate
        self.weight_torch = Config.get_mel_weight_torch(percent=1.0)[None, None, None, ...]

    def forward(self, mel, cuda=False):
        """
        :param non normalized mel spectrogram: [batchsize, 1, t-steps, n_mel]
        :return: [batchsize, 1, samples]
        """
        assert mel.size()[-1] == 128

        mel = try_tensor_cuda(mel, cuda=cuda)
        self.weight_torch = self.weight_torch.type_as(mel)
        mel = mel / self.weight_torch
        mel = tr_normalize(tr_amp_to_db(torch.abs(mel)) - 20.0)
        mel = tr_pre(mel[:, 0, ...])

        session = InferenceSession("models/voc.onnx", providers=["CPUExecutionProvider"])
        logits = session.run(["output"], {"input": mel.numpy()})
        result = from_numpy(logits[0])

        return result


if __name__ == "__main__":
    model = Vocoder(sample_rate=44100)
