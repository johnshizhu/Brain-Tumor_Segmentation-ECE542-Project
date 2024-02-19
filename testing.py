import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")
    print("Device:", device)
    print("Device index:", device.index)
    print("Device type:", device.type)
    print("Device name:", torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")