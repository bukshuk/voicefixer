from timeit import default_timer as timer
from voicefixer import VoiceFixer

voicefixer = VoiceFixer()

length = 68

base_dir = "audio"
file_name = f"zmm-{length}_ambe"

start_time = timer()

voicefixer.restore(input=f"{base_dir}/{file_name}.wav", output=f"{base_dir}/{file_name}_enhanced.wav", cuda=False)

end_time = timer()

total = end_time-start_time

print(f"{total:.2f} (sec). RTF: {total / length:.2f}")