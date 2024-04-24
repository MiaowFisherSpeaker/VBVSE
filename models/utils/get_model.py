import torch
from models import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(pt_path=config.best_checkpoint, map_location=device):
    checkpoint = torch.load(pt_path,map_location=map_location)
    model = checkpoint['model']
    return checkpoint["epoch"], model
