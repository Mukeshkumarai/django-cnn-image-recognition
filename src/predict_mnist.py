import argparse
import torch
from PIL import Image
from torchvision import transforms
from models.cnn import SimpleCNN
from models.dnn import DNNMNIST


def load_image(path: str):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "dnn"], default="cnn")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "cnn":
        model = SimpleCNN()
    else:
        model = DNNMNIST()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    x = load_image(args.image).to(device)
    with torch.inference_mode():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    print("Predicted digit:", pred)


if __name__ == "__main__":
    main()


