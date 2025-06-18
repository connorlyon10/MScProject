from model.wav2vec2_setup import initialise_model
from utils.audio_processing import transcribe_audio

def main():
    # Initialize the Wav2Vec2 model
    model, processor = initialise_model()

    audio_file_path = "commonvoice0104-en\cv-corpus-21.0-delta-2025-03-14\en\clips\common_voice_en_41910499.mp3"

    # Transcribe the audio file
    detected_text = transcribe_audio(audio_file_path, model, processor)
    
    # Print the detected text
    print("Detected text:", detected_text)

if __name__ == "__main__":
    main()