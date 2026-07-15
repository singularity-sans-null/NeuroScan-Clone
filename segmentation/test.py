import torch

checkpoint = torch.load(
    "saved_models/last_undet.pth",
    map_location="cpu"
)

print(checkpoint.keys())

print(checkpoint.get("history"))