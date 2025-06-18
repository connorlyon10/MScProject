from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def initialise_model():
    model_name = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model