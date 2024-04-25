# 使用./indexs/Iq2.index 进行文搜图
# 对于每个query，返回top-5的图片
import faiss
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

from get_3_i2t_index import TextDataset, Tx
from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
from get_test_index import get_features, preprocess_T


def predict_q2(img_path="./data/q2/ImageData/",output_folder='./data/q2/',
               query_text_path='./data/q2/word_test.csv',result_folder = './excel/q2.csv'):
    # 返回excel格式
    # text_id(query) similarity_ranking result_image_id

    # 读取index
    Iq2_index = faiss.read_index("./indexs/Iq2.index")
    # 读取图片数据（处理后的）
    img_paths = pd.read_csv(output_folder + 'my_image.csv')['image_id'].tolist()


    # 读取模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model = get_model(pt_path="./pts/best_model0416.ckpt", map_location=device)

    # 读取query
    query_data = pd.read_csv(query_text_path)
    query_id = query_data["text_id"].tolist() # 到时候保存这个
    query_text = query_data["caption"].tolist() # 推理用这个

    # print(query_text)
    # 对于query,也使用Dataset的方式
    batch_size = 32
    query_dataset = TextDataset(query_text,transform=None)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 文搜图，备用一个无关的图片
    noUseimg = img_paths[0]
    # print("这里没问题")
    noUseimg = Image.open(noUseimg).convert("RGB")
    noUseimg = Tx(noUseimg)
    noUseimg = noUseimg.squeeze(1).to(device, non_blocking=True)

    # 初始化表头
    with open(result_folder, 'w', newline='', encoding='utf-8') as f:
        f.write('text_id,similarity_ranking,result_image_id\n')


    for i, (idx, caps) in enumerate(query_loader):
        # text是一个包含batch_size的tuple，元素为文本
        # 对text处理
        caps = preprocess_T(caps)
        caps = {k: v.to(device, non_blocking=True) for k, v in caps.items()}
        with torch.no_grad():
            _, text_code = model(noUseimg, caps, 1)
        print(text_code.shape)
        # 对于每个query，返回top-5的图片
        # 用Iq2_index进行搜索
        D, I = Iq2_index.search(text_code.cpu().numpy(), 5)
        # print(I.shape)
        # break
        # 保存结果 这里一次性找的是batch_size个query,所以I有batch_size行
        for j in range(I.shape[0]):
            result = pd.DataFrame({"text_id": [query_id[j]] * 5,
                                   "similarity_ranking": [1, 2, 3, 4, 5],
                                   "result_image_id": [img_paths[k].split('/')[-1] for k in I[j]]})
            result.to_csv(result_folder, mode='a', index=False, header=False)


if __name__ == '__main__':
    predict_q2()