import unittest

from timeit import default_timer as timer

import soundfile
import numpy as np

from voicefixer import VoiceFixer


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self._vf = VoiceFixer()

    def test_12s(self):
        self.process(12)

    def test_19s(self):
        self.process(19)

    def test_20s(self):
        self.process(20)

    def test_68s(self):
        self.process(68)

    def process(self, index):
        name = f"zmm-{index}_ambe"
        in_file_name = f"{name}.wav"
        out_file_name = f"{name}_vf.wav"

        start_time = timer()
        self._vf.restore(input=f"audio/{in_file_name}", output=f"audio/{out_file_name}")
        end_time = timer()

        rtf = (end_time - start_time) / index

        self.assertGreater(rtf, 0.5)
        self.assertLess(rtf, 2)

        self.check_wav(out_file_name)

    def check_wav(self, file_name):
        expected_file_path = f"test/check/{file_name}"
        current_file_path = f"audio/{file_name}"

        expected_signal, expected_sr = soundfile.read(expected_file_path)
        current_signal, current_sr = soundfile.read(current_file_path)

        self.assertEqual(expected_sr, current_sr)
        self.assertEqual(len(expected_signal), len(current_signal))

        eps = 0.0005
        signal_diff = np.mean(np.abs(expected_signal - current_signal))
        self.assertLess(signal_diff, eps)


if __name__ == "__main__":
    unittest.main()
