import torch
import soundfile as sf
import librosa
from pathlib import Path
import re

from src.utils.SpectrogramExtractor import SpectrogramExtractor
from src.model import ConvCount, config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load the model from a .pt file
def load_model(weight_path):
    model = ConvCount(**config)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model



# split a wav into 1s clips
def save_wav_as_clips(wav_path: Path, out_dir: Path, length: float = 1.0):
    """
    Splits a `.wav` file into clips of duration `length` and saves them to the `out_dir`.
    """
    wav_path = Path(wav_path)
    out_dir = Path(out_dir)


    out_dir.mkdir(parents=True, exist_ok=True)

    # Formatting for file names
    speaker = wav_path.parents[2].name
    session_folder = wav_path.parents[1].name
    m = re.search(r"(session\d+)", session_folder)
    session = m.group(1) if m else session_folder


    # Split the audio into clips
    clips = split_wav_into_clips(wav_path, clip_dur=length, sr=16000)


    # Save clips
    for idx, clip in enumerate(clips[1:-1], start=1):                              # <--- This trims the first and last second of audio. Refactor for prod
        fname = f"{speaker}_{session}_clip{idx}.wav"
        sf.write(out_dir / fname, clip, 16000)



# create spectrograms from a directory of wav files
def clips_to_specs(clips_dir, specs_dir):
    extractor = SpectrogramExtractor()

    clips_dir = Path(clips_dir)
    specs_dir = Path(specs_dir)
    specs_dir.mkdir(parents=True, exist_ok=True)

    # Convert all clips to spectrograms
    for wav_path in clips_dir.glob("*.wav"):

        spec = extractor(str(wav_path))   # Returns a tensor shape: [1, n_mels, time_frames]
        spec = spec.squeeze(0)  # Remove the leading channel dim. Now [n_mels, time_frames]
        

        out_path = specs_dir / wav_path.with_suffix(".pt").name
        torch.save(spec, out_path) # save as a PyTorch tensor



# make prediction given a spectrogram
def predict(spectrogram_path, model):
    x = load_spectrogram(spectrogram_path)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
    return pred



# region Helpers

# helper for save_wav_as_clips
def split_wav_into_clips(wav_path, clip_dur=1.0, sr=16000):
    wav, _ = librosa.load(wav_path, sr=sr, mono=True) # librosa handles mono conversion for us
    clip_len = int(clip_dur * sr)
    num_clips = len(wav) // clip_len

    clips = [
        wav[i * clip_len : (i + 1) * clip_len]
        for i in range(num_clips)
    ]
    return clips



# Helper for predict()
def load_spectrogram(path):
    tensor = torch.load(path, map_location=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
    return tensor

# endregion


# if running as script:

# if __name__ == "__main__":
#     model = load_model("src/model/SpeakerCountCNN_v0.01.pt")
#     path = "data/spectrograms/0L_session0_clip3.pt"
#     result = predict(model, path)
#     print(f"Predicted class: {result}")
