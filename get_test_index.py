import json
import pandas as pd
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor
from PIL import Image
from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
from typing import Union

batch_size = 32


def my_model(_pt_path="./pts/best_model0419.ckpt"):
    model = get_model(pt_path=_pt_path)
    print("model加载完毕")
    return model


def get_dataset_data(json_path, all_data_path="./data/ImageWordData.csv"):
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.read_csv(all_data_path, encoding="utf-8")
    return data, df


def preprocess_I(image_paths):
    """
    处理图像
    这里只做单张处理
    """
    # 对于字符串传入得做如下处理
    ls = []
    imgs = []
    # 检测image_paths是否是列表
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    for image_path in image_paths:
        # img = Image.open(image_path).convert("RGB")
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise e

        def Tx(x):
            processor = AutoImageProcessor.from_pretrained(
                'google/vit-base-patch16-224-in21k')  # 处理得到的是字典，取pixel_values
            Txx = processor(x, return_tensors="pt")
            # 确保 processor 返回的是一个字典，并且包含 'pixel_values' 键
            if "pixel_values" not in Txx:
                raise KeyError("The processor did not return 'pixel_values'")
            return Txx["pixel_values"]

        img = Tx(img)
        # print(type(img))
        imgs.append(img)

    return torch.stack(imgs).squeeze(1)


def preprocess_T(caps: Union[str, list]):
    """
    处理文本
    这里只做单文本处理，虽然可以传列表。
    """
    caps = config.tokenizer(caps, padding=True, truncation=False, return_tensors='pt')  # 这里对于一批数据需要自动padding
    return caps


def get_features(model, img_paths, captions):
    """
    这里imgs和caps好像可以不对等
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = preprocess_I(img_paths).to(device, non_blocking=True)
    caps = preprocess_T(caps=captions)
    caps = {k: v.to(device, non_blocking=True) for k, v in caps.items()}
    with torch.no_grad():
        image_code, text_code = model(imgs, caps, cap_lens=1)
    return image_code, text_code


def main(MODEL_PATH="./pts/best_model0419.ckpt", getWhat="T"):
    import logging

    logging.basicConfig(filename="get_test_index.log",
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode="a",
                        level=logging.INFO)
    logging.info("开始运行")

    my_test_data, df = get_dataset_data(json_path="test_data.json")
    print("加载数据成功")
    logging.info("加载数据成功")
    captions, img_paths = [], []
    for i, caption in enumerate(my_test_data["CAPTIONS"]):
        captions.append(caption)

    for i, img_path in enumerate(my_test_data["IMAGES"]):
        img_paths.append(img_path)
    print("数据全部读取")
    logging.info("数据全部读取")
    model = my_model(MODEL_PATH)
    print("模型加载成功")
    logging.info("模型加载成功")

    if getWhat == "I":
        # 文搜图，保存图向量到I_index
        I_index = faiss.IndexFlatIP(1024)
        I_index_map = faiss.IndexIDMap(I_index)

        print("创建Image的index成功")
        logging.info("创建Image的index成功")

        cut = range(0, len(my_test_data["IMAGES"]), batch_size)

        for i in tqdm(cut):
            end_id = min(i + batch_size, len(my_test_data["IMAGES"]))

            image_code, _ = get_features(model=model, img_paths=img_paths[i:end_id],
                                         captions="")  # 直接算内存不足。存到向量数据库里。
            I_index_map.add_with_ids(image_code.cpu().numpy(), np.arange(i, end_id))
            print(image_code.shape)

            print(f"I-{i}添加成功")
            logging.info(f"I-{i}添加成功")

        faiss.write_index(I_index_map, "I.index")
        print("Image的index全部写入成功")
        logging.info("Image的index全部写入成功")
    if getWhat == "T":
        #图搜文，保存文向量到T_index
        T_index = faiss.IndexFlatIP(1024)
        T_index_map = faiss.IndexIDMap(T_index)

        print("创建Text的index成功")
        logging.info("创建Text的index成功")
        cut = range(0, len(my_test_data["CAPTIONS"]), batch_size)
        for i in tqdm(cut):
            end_id = min(i + batch_size, len(my_test_data["CAPTIONS"]))
            _, text_code = get_features(model=model, img_paths=img_paths[0],
                                        captions=captions[i:end_id])
            T_index_map.add_with_ids(text_code.cpu().numpy(), np.arange(i, end_id))
            print(text_code.shape)
            print(f"T-{i}添加成功")
            logging.info(f"T-{i}添加成功")
        faiss.write_index(T_index_map, "T.index")
        print("Text的index全部写入成功")
        logging.info("Text的index全部写入成功")

    print("程序终止")
    logging.info("程序终止")


if __name__ == "__main__":
    # MODEL_PATH填入模型路径，getWhat填入"I"或"T"，分别获取图向量和文向量对应的.index，可分别用于文搜图和图搜文
    main(MODEL_PATH="./pts/best_model0416.ckpt", getWhat="T")
