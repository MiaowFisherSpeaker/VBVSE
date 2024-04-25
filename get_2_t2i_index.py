# 文搜图, 保存I.index
# 对附件2图像进行处理
# 得到第二题搜索的T.index
import os

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor

from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
from get_test_index import get_features, TestDataset, preprocess_T
from torch.utils.data import Dataset, DataLoader

processor = AutoImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k')  # 处理得到的是字典，取pixel_values
def Tx(x):
    Txx = processor(x, return_tensors="pt")
    # 确保 processor 返回的是一个字典，并且包含 'pixel_values' 键
    if "pixel_values" not in Txx:
        raise KeyError("The processor did not return 'pixel_values'")
    return Txx["pixel_values"]
class ImageDataset(Dataset):
    def __init__(self, img_paths,transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return idx, img

# 数据集处理
def create_2_dataset(img_path="./data/q2/ImageData/", output_folder='./data/q2/'):
    # 通过image_data.csv读取image_id
    data = pd.read_csv("./data/q2/image_data.csv")

    def get_full_path(image_id):
        return img_path + image_id

    image_paths = data['image_id'].apply(get_full_path)
    # print(type(image_paths)) # Series
    # 检测图片是否损坏
    # """ 以下是发现的模型不能处理的图片（异常图片，高度为1）:
    # Image14001007 - 5658
    # Image14001007 - 6471
    # Image14001009 - 5865
    # Image14001010 - 7319
    # Image14001011 - 2253
    # Image14001013 - 4546
    # """
    if os.path.exists(output_folder + 'my_image.csv'):
        img_paths = pd.read_csv(output_folder + 'my_image.csv')['image_id'].tolist()
    else:
        img_paths = []
        print(f"处理前{len(image_paths)}")
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                if img.size[1] == 1:
                    print(f"{path}高度不适配模型")
                else:
                    img_paths.append(path)
            except Exception as e:
                print(f"捕获异常数据位于:{path}",e)
            if i%10000 == 0:
                print(f"已检测{i}张图片")
        print("处理后",len(img_paths))
        # 可以作为index的图片信息保存到json
        img_paths = pd.DataFrame(img_paths, columns=['image_id'])
        img_paths.to_csv(output_folder + 'my_image.csv', index=False)


    # 保存T.index到ouput_folder
    print(type(img_paths)) # list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model = get_model(pt_path="./pts/best_model0416.ckpt", map_location=device)
    print("model加载完毕")

    # 保存Iq2.index
    I_index = faiss.IndexFlatIP(1024)
    Iq2_index_map = faiss.IndexIDMap(I_index)
    batch_size = 32

    # 这里直接处理很慢，需要借助torch的Dataset和DataLoader
    I_dataset = ImageDataset(img_paths, transform = Tx)
    I_loader = DataLoader(dataset=I_dataset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)

    # cut = range(0, len(img_paths), batch_size)
    # for i in tqdm(cut):
    #     end_id = min(i + batch_size, len(img_paths))
    #
    #     image_code, _ = get_features(model=model, img_paths=img_paths[i:end_id],
    #                                  captions="")
    #     Iq2_index_map.add_with_ids(image_code.cpu().numpy(), np.arange(i, end_id))
    #     print(image_code.shape)
    #     print(f"Iq2-{i+1}添加成功")

    print("图像加载完成")
    caps = preprocess_T(caps="")
    caps = {k: v.to(device, non_blocking=True) for k, v in caps.items()}

    for i,(idx,img) in enumerate(tqdm(I_loader)):
        img = img.squeeze(1).to(device, non_blocking=True)
        with torch.no_grad():
            image_code, _ = model(img, caps, 1)
        Iq2_index_map.add_with_ids(image_code.cpu().numpy(), idx.numpy())
        print(image_code.shape)
        print(f"Iq2-{i+1}添加成功")


    faiss.write_index(Iq2_index_map, output_folder + "Iq2.index")

if __name__ == '__main__':
    create_2_dataset()
