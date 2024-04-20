import json
from collections import defaultdict

import faiss
from tqdm import tqdm
from get_test_index import get_features
from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP


def calculate_recall_at_k(model_path="./pts/best_model0419.ckpt", mode="I2T", data_json_path="test.json",
                          I_index_path=None, T_index_path=None, k=5):
    """
    :params
    mode: str I2T图搜文 | T2I文搜图
    data_json_path: str
    """
    with open(data_json_path, "r") as f:
        data = json.load(f)
    captions, img_paths = [], []
    for i, caption in enumerate(data["CAPTIONS"]):
        captions.append(caption)

    for i, img_path in enumerate(data["IMAGES"]):
        img_paths.append(img_path)

    model = get_model(model_path)

    if mode == "T2I":  # 文搜图，从Image_index里找topk
        I_index_map = faiss.read_index(I_index_path)
        count = 0
        for i in range(len(tqdm(captions))):
            # print(i,captions[i])
            # print(img_paths[0:1],captions[0])
            # print(os.getcwd())
            _, text_code = get_features(model=model, img_paths=img_paths[0],
                                        captions=captions[i])
            text_code = text_code.cpu().numpy()
            # print(text_code.shape)

            D, I = I_index_map.search(text_code, k)
            # print(i,captions[i],I[0])

            if i in I[0]:
                count += 1
        return count / len(captions)
    if mode == "I2T":  # 图搜文，从Text_index里找topk
        T_index_map = faiss.read_index(T_index_path)
        count = 0
        for i in range(len(tqdm(img_paths))):
            # print(i,captions[i])
            # print(img_paths[0:1],captions[0])
            # print(os.getcwd())
            img_code, _ = get_features(model=model, img_paths=img_paths[i],
                                       captions="")
            img_code = img_code.cpu().numpy()

            D, I = T_index_map.search(img_code, k)
            if i in I[0]:
                count += 1
        return count / len(img_paths)

# 这个效率很低啊，建议自己写吧
if __name__ == "__main__":
    import logging

    # 计算best_model01416与0419的Recall@k,k取1,5,10,100，并记录日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='calculate_recall_at_k.log',
                        filemode='a+')
    # 0416
    model_path0416 = "./pts/best_model0416.ckpt"
    I_index_path0416 = "./indexs/I0416.index"
    T_index_path0416 = "./indexs/T0416.index"
    # 0419
    model_path0419 = "./pts/best_model0419.ckpt"
    I_index_path0419 = "./indexs/I0419.index"
    T_index_path0419 = "./indexs/T0419.index"

    data_json_path = "test_data.json"

    k_list = [1, 5, 10, 100]

    recall_dic = {}
    recall_dic["0416"] = defaultdict(list)
    #{"0416":{"I2T":[],"T2I":[]},"0419":{"I2T":[],"T2I":[]


    # 计算0416 Recall@k
    for k in k_list:
        logging.info(f"开始计算0416 I2T的Recall@{k}")
        # I2T
        recall = calculate_recall_at_k(
            model_path=model_path0416,
            mode="I2T",
            data_json_path=data_json_path,
            I_index_path=I_index_path0416,
            T_index_path=T_index_path0416,
            k=k
        )
        recall_dic["0416"]["I2T"].append(recall)
        logging.info(f"0416 I2T Recall@{k}:{recall}")

        logging.info(f"开始计算0416 T2I的Recall@{k}")

        # T2I
        recall = calculate_recall_at_k(
            model_path=model_path0416,
            mode="T2I",
            data_json_path=data_json_path,
            I_index_path=I_index_path0416,
            T_index_path=T_index_path0416,
            k=k
        )
        recall_dic["0416"]["T2I"].append(recall)
        logging.info(f"0416 T2I Recall@{k}:{recall}")

    for k in k_list:
        logging.info(f"开始计算0419 I2T的Recall@{k}")
        # I2T
        recall = calculate_recall_at_k(
            model_path=model_path0419,
            mode="I2T",
            data_json_path=data_json_path,
            I_index_path=I_index_path0419,
            T_index_path=T_index_path0419,
            k=k
        )
        recall_dic["0419"]["I2T"].append(recall)
        logging.info(f"0419 I2T Recall@{k}:{recall}")

        logging.info(f"开始计算0419 T2I的Recall@{k}")

        # T2I
        recall = calculate_recall_at_k(
            model_path=model_path0419,
            mode="T2I",
            data_json_path=data_json_path,
            I_index_path=I_index_path0419,
            T_index_path=T_index_path0419,
            k=k
        )
        recall_dic["0419"]["T2I"].append(recall)
        logging.info(f"0419 T2I Recall@{k}:{recall}")
    logging.info(str(recall_dic))