import torch
from src.model.model import SpeakerCountCNN, input_height, input_width


def load_model(weight_path):
    model = SpeakerCountCNN(input_height, input_width)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def load_spectrogram(path):
    tensor = torch.load(path, map_location=torch.device("cpu"))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
    return tensor


def predict(model, spectrogram_path):
    x = load_spectrogram(spectrogram_path)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
    return pred


if __name__ == "__main__":
    model = load_model("src/model/SpeakerCountCNN_v0.01.pt")
    path = "data/spectrograms/0L_session0_clip3.pt"
    result = predict(model, path)
    print(f"Predicted class: {result}")
