import torchaudio

from speechsep.dataset import get_audio_path
from speechsep.plotting import plot_waveform, plot_specgram

# %%
filename = get_audio_path("8842-304647-0012_3752-4943-0024", "dev")

metadata = torchaudio.info(filename)
print(metadata)

# %%
waveform, sample_rate = torchaudio.load(filename)

plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)
