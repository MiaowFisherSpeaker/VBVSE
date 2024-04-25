# 图搜文，保存T.index
# 对附件3文本进行处理
# 1. 根据文本划分batch_size的数据集
# 2. 得到第三题搜索的I.index
import faiss
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoImageProcessor

from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
from get_test_index import get_features, preprocess_T, preprocess_I

processor = AutoImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k')  # 处理得到的是字典，取pixel_values
def Tx(x):
    Txx = processor(x, return_tensors="pt")
    # 确保 processor 返回的是一个字典，并且包含 'pixel_values' 键
    if "pixel_values" not in Txx:
        raise KeyError("The processor did not return 'pixel_values'")
    return Txx["pixel_values"]
class TextDataset(Dataset):
    def __init__(self, captions:list,transform):
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        cap = self.captions[idx]
        # 不能保证长度相同，不做处理
        # if self.transform:
        #     cap = self.transform(cap)

        return idx, cap

def create_3_dataset(img_path="./data/q3/ImageData/",output_folder="./data/q3/"):
    # 通过image_test.csv读取image_id
    query_data = pd.read_csv("./data/q3/image_test.csv")

    def get_full_path(image_id):
        return img_path + image_id

    image_paths = query_data['image_id'].apply(get_full_path)
    # print(image_paths)

    # img_paths = []
    # print(f"处理前{len(image_paths)}")
    # for i, path in enumerate(image_paths):
    #     try:
    #         img = Image.open(path)
    #         if img.size[1] == 1:
    #             print(f"{path}高度不适配模型")
    #         else:
    #             img_paths.append(path)
    #     except Exception as e:
    #         print(f"捕获异常数据位于:{path}",e)
    #     if i%1000 == 0:
    #         print(f"已检测{i}张图片")
    # print("处理后",len(img_paths))

    # 测试集完好！
    img_paths = image_paths
    # 读取文本数据
    data = pd.read_csv("./data/q3/word_data.csv")
    captions = data["caption"].tolist()
    # print(captions[:5])

    # 保存Tq3.index到ouput_folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model = get_model(pt_path="./pts/best_model0416.ckpt", map_location=device)
    print("model加载完毕")

    T_index = faiss.IndexFlatIP(1024)
    Tq3_index_map = faiss.IndexIDMap(T_index)
    batch_size = 32

    # 也使用DataLoader
    # cut = range(0, len(captions), batch_size)
    # for i in tqdm(cut):
    #     end_id = min(i + batch_size, len(captions))
    #     _, text_code = get_features(model=model, img_paths=img_paths[0],
    #                                 captions=captions[i:end_id])
    #     Tq3_index_map.add_with_ids(text_code.cpu().numpy(), np.arange(i, end_id))
    #     print(text_code.shape)
    #     print(f"Tq3-{i+1}添加成功")
    T_dataset = TextDataset(captions,transform=preprocess_T)
    T_loader = DataLoader(dataset=T_dataset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)

    noUseimg = img_paths[0]
    print("这里没问题")
    noUseimg = Image.open(noUseimg).convert("RGB")
    noUseimg = Tx(noUseimg)
    noUseimg = noUseimg.squeeze(1).to(device, non_blocking=True)

    for i,(idx,caps) in enumerate(tqdm(T_loader)):
        # print(idx,caps)
        caps = preprocess_T(caps)
        caps = {k: v.to(device, non_blocking=True) for k, v in caps.items()}
        with torch.no_grad():
            _, text_code = model(noUseimg, caps, 1)
        Tq3_index_map.add_with_ids(text_code.cpu().numpy(), idx.numpy())
        print(text_code.shape)
        print(f"Tq3-{i+1}添加成功")
    faiss.write_index(Tq3_index_map, output_folder + "Tq3.index")
if __name__ == '__main__':
    create_3_dataset()