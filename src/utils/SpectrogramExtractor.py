# Extracts a spectrogram from an audio file
import torchaudio
import torchaudio.transforms as T

class SpectrogramExtractor:
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, n_mels=64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

    def __call__(self, filepath):
        waveform, sr = torchaudio.load(filepath)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        return log_mel_spec  # shape: (1, n_mels, time_frames)
    


# Creates a pytorch dataset 
# import os
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# class SpectrogramDataset(Dataset):
#     def __init__(self, csv_file, data_dir):

#         self.data = pd.read_csv(csv_file)
#         self.data_dir = data_dir


#     def __len__(self):
    
#         return len(self.data)
    

#     def __getitem__(self, idx):
    
#         row = self.data.iloc[idx]
#         tensor_path = os.path.join(self.data_dir, row['spectrogram'])
#         spectrogram = torch.load(tensor_path).unsqueeze(0).float();  # shape: [1, H, W]
#         label = int(row['speaker_count'])
#         return spectrogram, label