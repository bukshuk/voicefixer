import soundfile
import numpy as np

from timeit import default_timer as timer
from voicefixer import VoiceFixer


vf = VoiceFixer()


def test_12s():
    process(12)


def test_19s():
    process(19)


def test_20s():
    process(20)


def test_68s():
    process(68)


def process(index):
    name = f"zmm-{index}_ambe"
    in_file_name = f"{name}.wav"
    out_file_name = f"{name}_vf.wav"

    start_time = timer()
    vf.restore(input=f"audio/{in_file_name}", output=f"audio/{out_file_name}")
    end_time = timer()

    rtf = (end_time - start_time) / index

    assert rtf > 0.5 and rtf < 3.5

    if rtf > 1.5:
        print(f"RTF: {rtf}")

    check_wav(out_file_name)


def check_wav(file_name):
    expected_file_path = f"tests/check/{file_name}"
    current_file_path = f"audio/{file_name}"

    expected_signal, expected_sr = soundfile.read(expected_file_path)
    current_signal, current_sr = soundfile.read(current_file_path)

    assert expected_sr == current_sr
    assert len(expected_signal) == len(current_signal)

    eps = 0.0005
    signal_diff = np.mean(np.abs(expected_signal - current_signal))

    assert signal_diff < eps
