import json
from collections import Counter
import time

import faiss
from jieba import posseg

from get_test_index import get_features
import pandas as pd

# 模型必须要的
from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config

def cut_text(text):
    words = posseg.cut(text)
    nouns = []
    for word, flag in words:
        if flag == 'n' or flag == 'ns' or flag == 'vn':  # 名词、专有、动名词
            nouns.append(word)
    # print("分词结果",nouns)
    return nouns
def search_by_word(my_test_data,model,_index,noun,k=15):
    # 名词搜图top15
    img_paths = my_test_data["IMAGES"]
    _, text_code = get_features(model=model, img_paths=img_paths[0],captions=noun)
    _, I = _index.search(text_code.cpu().numpy(), k)
    return I[0]
def search_by_split(my_test_data,model,_index,caption_id):
    my_caption = my_test_data["CAPTIONS"][caption_id]
    true_img_path = my_test_data["IMAGES"][caption_id]
    # 分词
    words = cut_text(my_caption)

    word_search_results = [search_by_word(my_test_data,model,_index,word) for word in words]

    # 找公共元素
    if not word_search_results:  # 如果word_search_results为空，直接返回空集合
        common_indices = set()
    else:
        common_indices_sets = map(set, word_search_results)
        common_indices = set.intersection(*common_indices_sets)  # 求交集

    # 评分机制，统计每个索引在结果中出现的次数和位置
    score_dict = Counter()
    # 将共同元素加入到评分机制中，如果它们当前的分数不是最高的
    for index in common_indices:
        if index not in score_dict or score_dict[index] <= score_dict.most_common(1)[0][1]:
            score_dict[index] = score_dict.most_common(1)[0][1] + 1

    # 根据评分机制选取前5个索引，但要确保包含共同元素
    top5_indices = score_dict.most_common(5)

    if len(top5_indices) < 5:
        # 对整个文本进行搜索，并将结果补足到top5中
        additional_search_results = search_by_word(my_test_data,model,_index,my_caption, k=5) # word传入整个文本,指定k=5
        additional_indices = set(additional_search_results) - set([index for index, _ in top5_indices])
        additional_indices = sorted(additional_indices, key=lambda x: additional_search_results.index(x))[
                             :5 - len(top5_indices)]
        top5_indices.extend(additional_indices)

    return [index for index, _ in top5_indices]


def main(mode,excel_filePath,last_checkpoint=None):
    dataset_name = "泰迪杯2024B"
    json_path = "jsons/泰迪杯2024B/test_data.json"
    with open(json_path, "r") as f:
        my_test_data = json.load(f)


    pt_path = "pts/best_model0416.ckpt"
    _,model = get_model(pt_path=pt_path)

    captions, img_paths = [], []
    for i, caption in enumerate(my_test_data["CAPTIONS"]):
        captions.append(caption)

    for i, img_path in enumerate(my_test_data["IMAGES"]):
        img_paths.append(img_path)

    data = []
    I_index_map = faiss.read_index("indexs/I0416.index")

    # if last_checkpoint:
    #     with

    if mode=="raw":
        for k,my_num in enumerate(range(len(my_test_data["CAPTIONS"]))):
            my_caption = captions[my_num]
            time1 = time.time()
            _, text_code = get_features(model=model, img_paths=img_paths[0],
                                        captions=my_caption)
            text_code = text_code.cpu()
            D, I = I_index_map.search(text_code, 5)
            top5_id_ls = I[0]
            time2 = time.time()
            for i, img_id in enumerate(top5_id_ls):
                data.append({
                    "caption_id": my_num,
                    "caption": my_caption,
                    "img_id": img_id,
                    "img_path": img_paths[img_id],
                    "rank": i,
                    "time_cost": time2-time1
                })
            print(f"第{k+1}个caption搜索完成")
            last_checkpoint = k
            with open(f"./excel/testResult/{excel_filePath.split('.')[0]}/last_checkpoint.txt", "w") as f:
                f.write(str(last_checkpoint))


    if mode == "split":
        for k,my_num in enumerate(range(len(my_test_data["CAPTIONS"]))):
            my_caption = my_test_data["CAPTIONS"][my_num]
            time1 = time.time()
            top5_id_ls = search_by_split(my_test_data,model,I_index_map,my_num)
            time2 = time.time()
            for i, img_id in enumerate(top5_id_ls):
                data.append({
                    "caption_id": my_num,
                    "caption": my_caption,
                    "img_id": img_id,
                    "img_path": img_paths[img_id],
                    "rank": i+1,
                    "time_cost": time2-time1
                })
            print(f"第{k}个caption搜索完成")
            last_checkpoint = k
            with open(f"./excel/testResult/{excel_filePath.split('.')[0]}/last_checkpoint.txt", "w") as f:
                f.write(str(last_checkpoint))

    df = pd.DataFrame(data)
    # excel_filePath = "./excel/testResult/split_0416.xlsx"
    df.to_excel(excel_filePath, index=False)

if __name__ == '__main__':
    # main(
    #     # 选"raw","split"
    #     mode = "split",
    #     excel_filePath = "./excel/testResult/split_0416.xlsx"
    # )
    main(
        # 选"raw","split"
        mode = "raw",
        excel_filePath = "./excel/testResult/raw_0416.xlsx"
    )