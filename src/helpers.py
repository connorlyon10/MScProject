from IPython.display import display
import IPython.display as ipd

def load_audio(wav_files):
    """
    Display audio files as buttons.
    
    Args:        
        wav_files: path or list of paths to audio files.
    """

    if not isinstance(wav_files, list):
        wav_files = [wav_files]

    for wav in wav_files:
        display(ipd.Audio(wav))
        print(f"Above: {wav}\n")