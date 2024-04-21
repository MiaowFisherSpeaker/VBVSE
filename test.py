import json

from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
import torch

if __name__ == '__main__':
    model = get_model('pts/best_model0416.ckpt')
    print("model加载完毕")
    data_dir = './'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)

    with open("jsons/泰迪杯2024B/test_data.json", "r") as f:
        data = json.load(f)
    print(data)