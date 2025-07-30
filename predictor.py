from model.model import captcha_model, model_conv, model_resnet
from utils.arg_parsers import predict_arg_parser
from data.dataset import str_to_vec, lst_to_str

import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
        transforms.Resize((60, 160)),  # Ensure correct size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(args):
    model = captcha_model.load_from_checkpoint(args.ckpt, model=model_resnet())
    model.eval()
    
    # Open image and convert to RGB to handle different formats
    img = Image.open(args.input).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    y = model(img)
    y = y.permute(1, 0, 2)
    pred = y.argmax(dim=2)

    ans = lst_to_str(pred)
    print(f"Predicted CAPTCHA: {ans}")
    return ans


if __name__ == "__main__":
    args = predict_arg_parser()
    predict(args)
    

