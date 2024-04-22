import json

from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
from models import evaluate
import torch
import logging

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    epoch, model = get_model('pts/best_model0420.ckpt', map_location=device)
    print("model加载完毕")

    data_dir = "./"
    _, _, test_loader = mktrainval(data_dir, config.batch_size)
    print("数据加载完毕")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='my_logging.log',
                        filemode='a')
    print("日志挂载完成")

    r1_i2t, r1_t2i, r5_i2t, r5_t2i, r10_i2t, r10_t2i = evaluate(data_loader=test_loader, model=model,
                                                                batch_size=config.batch_size,
                                                                captions_per_image=config.captions_per_image)
    print(
        f"Epoch: {epoch}, \n I2T R@1: {r1_i2t}, T2I R@1: {r1_t2i}, \t I2T R@5: {r5_i2t}, T2I R@5: {r5_t2i}, \t I2T R@10: {r10_i2t}, T2I R@10: {r10_t2i}")
    logging.info(
        f"Epoch: {epoch}, \n I2T R@1: {r1_i2t}, T2I R@1: {r1_t2i}, \t I2T R@5: {r5_i2t}, T2I R@5: {r5_t2i}, \t I2T R@10: {r10_i2t}, T2I R@10: {r10_t2i}")