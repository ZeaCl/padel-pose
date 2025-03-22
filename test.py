import torch

if torch.backends.mps.is_available():
    print("MPS backend is available!")
    device = torch.device("mps")
else:
    print("MPS backend is not available.")
    device = torch.device("cpu")

x = torch.rand(2, 2, device=device)
print(x)