import torch


def get_model(pt_path):
    checkpoint = torch.load(pt_path)
    model = checkpoint['model']
    return model
