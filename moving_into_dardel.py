
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")
    # Add any GPU-accelerated logic here