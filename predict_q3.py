# 图搜文，从Tq3.index中搜索
# 对于每个query，返回top-5的文本id
import faiss
import pandas as pd
import torch
from torch.utils.data import DataLoader
from get_2_t2i_index import Tx, ImageDataset
from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
from get_test_index import get_features, preprocess_T


def predict_q3(data_path="./data/q3/ImageData/", output_folder="./data/q3",
               query_img_path="./data/q3/image_test.csv", result_folder="./excel/q3.csv"):
    # 返回excel格式
    # image_id, similarity_ranking , result_text_id

    Tq3_index = faiss.read_index("./indexs/Tq3.index")


    data = pd.read_csv("./data/q3/word_data.csv")
    captions_id = data["text_id"].tolist() # 从中得到text_id

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model = get_model(pt_path="./pts/best_model0416.ckpt", map_location=device)

    # # 一次处理一个query

    def get_full_path(image_id):
        return data_path + image_id

    query_data = pd.read_csv("./data/q3/image_test.csv")
    query_img_paths = query_data['image_id'].apply(get_full_path).tolist()
    #
    # for i, img_path in enumerate(img_paths):
    #     img_code,_ = get_features(model=model, img_paths=img_path,
    #                                captions="")
    #     D, I = Tq3_index.search(img_code.cpu().numpy(), 5)
    #     print(D,I)
    #     print(f"第{i+1}张图片搜索完成")
    #     # 保存结果
    #     result = pd.DataFrame({"image_id": [img_path] * 5,
    #                            "similarity_ranking": [1,2,3,4,5],
    #                            "result_text_id": I[0]})
    #     result.to_csv(f"./results/q3/{img_path}.csv", index=False)

    # 按批次处理
    batch_size = 32
    query_dataset = ImageDataset(query_img_paths, transform=Tx)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 初始化表头
    with open(result_folder, 'w', newline='', encoding='utf-8') as f:
        f.write('image_id,similarity_ranking,result_text_id\n')

    # 没有用的文本
    caps = preprocess_T(caps="")
    caps = {k: v.to(device, non_blocking=True) for k, v in caps.items()}

    for i, (idx, img) in enumerate(query_loader):
        img = img.squeeze(1).to(device, non_blocking=True)
        with torch.no_grad():
            img_code, _ = model(img, caps, 1)
        D, I = Tq3_index.search(img_code.cpu().numpy(), 5)

        print(img_code.shape)

        for j in range(I.shape[0]):
            result = pd.DataFrame({"image_id": [query_img_paths[j].split('/')[-1]] * 5,
                                   "similarity_ranking": [1, 2, 3, 4, 5],
                                   "result_text_id": [captions_id[k] for k in I[j]]})
            result.to_csv(result_folder, mode='a', header=False, index=False)

if __name__ == '__main__':
    predict_q3()