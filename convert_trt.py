import argparse
import os
import shutil
import torch
from pathlib import Path
from urllib.request import Request, urlopen
from torch2trt import torch2trt
from src.common import prepare, test


PRETRAINED_MODEL_URLS = {
    "2": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


def ensure_pretrained_model(model_file: Path, scale: str) -> None:
    """Download the official checkpoint when it is not already present."""
    if model_file.is_file():
        return

    model_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = model_file.with_suffix(model_file.suffix + ".part")
    request = Request(PRETRAINED_MODEL_URLS[scale], headers={"User-Agent": "RealESRGAN-TensorRT-Upscaler"})

    print(f"Pretrained model not found. Downloading {model_file.name}...")
    try:
        with urlopen(request, timeout=60) as response, partial_file.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        if partial_file.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded checkpoint is unexpectedly small; the download may have failed.")
        os.replace(partial_file, model_file)
        print(f"Downloaded pretrained model to {model_file}")
    except Exception:
        partial_file.unlink(missing_ok=True)
        raise

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Convert RealESRGAN model to TensorRT.")
    parser.add_argument("--height", type=int, default=512, help="Input height of the model")
    parser.add_argument("--width", type=int, default=512, help="Input width of the model")
    parser.add_argument("--pretrained_model", type=str, default="2", choices=["2", "4"],
                        help="Choose between pretrained RealESRGAN_x2plus or RealESRGAN_x4plus")
    parser.add_argument("--model_output_directory", type=str,
                        default="./src/models/tensorrt_model_output",
                        help="Output directory for saving TensorRT model")
    args = parser.parse_args()

    # Construct input size
    side = (args.width,args.height )

    # Check pretrained model path
    model_file = Path(f"./src/models/pretrained_models/RealESRGAN_x{args.pretrained_model}plus.pth")
    ensure_pretrained_model(model_file, args.pretrained_model)

    # Prepare model
    is_2x = False
    if args.pretrained_model.lower() == "2":
        is_2x = True
        
    model, x = prepare(side, str(model_file), is_2x)

    torch.cuda.empty_cache()
    with torch.no_grad():
        print("Converting to TensorRT...")
        model_trt = torch2trt(model, [x], fp16_mode=True)

    if model_trt is None or getattr(model_trt, "engine", None) is None:
        raise RuntimeError(
            "TensorRT did not create an engine. The GPU ran out of memory during conversion; "
            "retry with smaller --width and --height values, such as 512x512 or 256x256."
        )

    # Ensure output directory exists
    os.makedirs(args.model_output_directory, exist_ok=True)

    # Save model
    output_path = os.path.join(args.model_output_directory, f't2trt_{side[0]}_{side[1]}_x{args.pretrained_model}.trt')
    torch.save(model_trt.state_dict(), output_path)
    print(f"TensorRT model saved to {output_path}")

    # Test outputs
    y = test(x, model, "model")
    y_trt = test(x, model_trt, "model_trt")
    print("Error between original and TRT model:", torch.max(torch.abs(y - y_trt)))

if __name__ == "__main__":
    main()
