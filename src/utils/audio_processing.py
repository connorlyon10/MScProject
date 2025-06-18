from transformers import Wav2Vec2Processor
import soundfile as sf
from scipy.signal import resample

def load_audio(file_path):
    audio_input, sample_rate = sf.read(file_path)
    return audio_input, sample_rate

def prepare_audio(audio_input, sample_rate, target_sample_rate=16000):
    # Convert stereo to mono if necessary
    if len(audio_input.shape) > 1:
        audio_input = audio_input[:, 0]

    # Resample if sample rate isn't target
    if sample_rate != target_sample_rate:
        number_of_samples = int(len(audio_input) * target_sample_rate / sample_rate)
        audio_input = resample(audio_input, number_of_samples)

    return audio_input, target_sample_rate

def transcribe_audio(file_path, model, processor):
    audio_input, sample_rate = load_audio(file_path)
    processed_audio, target_sample_rate = prepare_audio(audio_input, sample_rate)
    input_values = processor(processed_audio, sampling_rate=target_sample_rate, return_tensors="pt").input_values
    logits = model(input_values).logits
    detected_text = processor.decode(logits[0], skip_special_tokens=True)
    return detected_text