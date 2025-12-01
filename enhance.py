import argparse

from timeit import default_timer as timer

from voicefixer import VoiceFixer


def get_length() -> int:
    ALLOWED_INDEXES = [12, 19, 20, 68]

    parser = argparse.ArgumentParser(description="Process a single index argument.")
    parser.add_argument(
        "index",
        type=int,
        choices=ALLOWED_INDEXES,
        help=f"The time index of the input file. Must be one of: ${ALLOWED_INDEXES}",
    )
    args = parser.parse_args()

    return args.index


def main(length: int):
    voicefixer = VoiceFixer()

    base_dir = "audio"
    file_name = f"zmm-{length}_ambe"

    start_time = timer()

    voicefixer.restore(input_path=f"{base_dir}/{file_name}.wav", output_path=f"{base_dir}/{file_name}_vf.wav")

    end_time = timer()

    total = end_time - start_time

    print(f"{total:.2f} (sec). RTF: {total / length:.2f}")


if __name__ == "__main__":
    main(get_length())
