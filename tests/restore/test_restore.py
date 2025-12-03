from pytest_bdd import scenarios, given, when, then, parsers

import librosa
import numpy as np

from timeit import default_timer as timer

from voicefixer import VoiceFixer


scenarios("restore.feature")


@given("the VoiceFixer model", target_fixture="vf")
def step_voicefixer():
    return VoiceFixer()


@when(parsers.parse("I restore recording with the {index:d}"), target_fixture="restored_file")
def step_restore(vf: VoiceFixer, index: int):
    name = f"zmm-{index}_ambe"
    in_file_name = f"{name}.wav"
    out_file_name = f"{name}_vf.wav"

    start_time = timer()
    vf.restore(input_path=f"audio/{in_file_name}", output_path=f"audio/{out_file_name}")
    end_time = timer()

    rtf = (end_time - start_time) / index

    assert rtf > 0.5 and rtf < 3.5

    if rtf > 1.5:
        print(f"RTF: {rtf}")

    return out_file_name


@then("I get the restored recording")
def step_check(restored_file: str):
    expected_file_path = f"tests/check/{restored_file}"
    current_file_path = f"audio/{restored_file}"

    expected_signal, _ = librosa.load(expected_file_path, sr=44100)
    current_signal, _ = librosa.load(current_file_path, sr=44100)

    assert len(expected_signal) == len(current_signal)

    eps = 1e-5
    signal_mse = np.mean((expected_signal - current_signal)**2)

    assert signal_mse < eps
