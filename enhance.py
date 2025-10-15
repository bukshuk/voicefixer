from voicefixer import VoiceFixer

voicefixer = VoiceFixer()

base_dir = "D:/tmp"
file_name = "zmm-01_ambe"

voicefixer.restore(input=f"{base_dir}/{file_name}.wav", output=f"{base_dir}/{file_name}_enhanced.wav", cuda=False)
