from timeit import default_timer as timer
from voicefixer import VoiceFixer

voicefixer = VoiceFixer()

base_dir = "D:/tmp"
file_name = "zmm-00_ambe"

start_time = timer()

voicefixer.restore(input=f"{base_dir}/{file_name}.wav", output=f"{base_dir}/{file_name}_enhanced.wav", cuda=False)

end_time = timer()

total = end_time-start_time

print(f"{total:.2f} (sec). RTF: {total / 20:.2f}")