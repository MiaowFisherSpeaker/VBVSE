import glob
import os.path
import time
from collections import Counter
from typing import Union

import faiss
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
from jieba import posseg

import json

import torch
from torch import cosine_similarity
from transformers import AutoImageProcessor
from get_test_index import preprocess_I,preprocess_T,get_features
from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
from preprocessCaptions import get_summary_text



# 缓存model
@st.cache_data
def my_model(_pt_path="best_model0416.ckpt"):
    _,model = get_model(pt_path=_pt_path)
    print("model加载完毕")
    return model


@st.cache_data
def get_dataset_data(json_path, dataset_name="泰迪杯2024B"):
    if dataset_name == "泰迪杯2024B":
        all_data_path = "./data/ImageWordData.csv"
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.read_csv(all_data_path, encoding="utf-8")
        return data, df
    if dataset_name == "flicker30k":
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.read_csv(f'./data/Flickr30k-CNA/test/flickr30k_cn_test.txt', sep='\t', header=None, names=['image_id', 'caption'])
        return data, df

def faiss_search(_index, code, k=5):
    D, I = _index.search(code, k)
    return D, I

# 针对文本搜图片的另一个策略：分词找名词主体
def cut_text(text):
    words = posseg.cut(text)
    nouns = []
    for word, flag in words:
        if flag == 'n' or flag == 'ns' or flag == 'vn':  # 名词、专有、动名词
            nouns.append(word)
    print("分词结果",nouns)
    return nouns
def search_by_word(my_test_data,model,_index,noun,k=15):
    # 名词搜图top15
    img_paths = my_test_data["IMAGES"]
    _, text_code = get_features(model=model, img_paths=img_paths[0],captions=noun)
    _, I = _index.search(text_code.cpu().numpy(), k)
    return I[0]


def search_by_split(my_test_data,model,_index,caption_id):
    # my_caption = my_test_data["CAPTIONS"][caption_id]
    # true_img_path = my_test_data["IMAGES"][caption_id]
    # # 分词
    # words = cut_text(my_caption)
    #
    # word_search_results = [search_by_word(my_test_data,model,_index,word) for word in words]
    #
    # # 找公共元素
    # if not word_search_results:  # 如果word_search_results为空，直接返回空集合
    #     common_indices = set()
    # else:
    #     common_indices_sets = map(set, word_search_results)
    #     common_indices = set.intersection(*common_indices_sets)  # 求交集
    #
    # # 评分机制，统计每个索引在结果中出现的次数和位置
    # score_dict = Counter()
    # # 将共同元素加入到评分机制中，如果它们当前的分数不是最高的
    # for index in common_indices:
    #     if index not in score_dict or score_dict[index] <= score_dict.most_common(1)[0][1]:
    #         score_dict[index] = score_dict.most_common(1)[0][1] + 1
    #
    # # 根据评分机制选取前5个索引，但要确保包含共同元素
    # top5_indices = score_dict.most_common(5)
    #
    # if len(top5_indices) < 5:
    #     # 对整个文本进行搜索，并将结果补足到top5中
    #     additional_search_results = search_by_word(my_test_data,model,_index,my_caption, k=5) # word传入整个文本,指定k=5
    #     additional_search_results = np.array(additional_search_results)
    #     additional_indices_set = set(additional_search_results)
    #     additional_indices = additional_indices_set - set([index for index, _ in top5_indices])
    #     unique_additional_indices, indices = np.unique(additional_search_results, return_inverse=True)
    #     sorted_indices = np.argsort(-np.bincount(indices))
    #     additional_indices_to_add = unique_additional_indices[sorted_indices][:5 - len(top5_indices)]
    #     original_indices_to_add = np.ndarray.tolist(np.take(indices, np.where(indices == additional_indices_to_add)[0]))
    #     top5_indices.extend(original_indices_to_add)
    #     top5_indices = list(set(top5_indices))
    #     top5_indices = top5_indices[:5]
    my_caption = my_test_data["CAPTIONS"][caption_id]
    true_img_path = my_test_data["IMAGES"][caption_id]
    # 分词
    words = cut_text(my_caption)
    word_search_results = [search_by_word(my_test_data, model, _index, word) for word in words]

    # 找公共元素
    common_indices = set.intersection(*map(set, word_search_results))
    # 评分机制，统计每个索引在结果中出现的次数和位置
    score_dict = Counter()
    for i, indices in enumerate(word_search_results):
        for index in indices:
            score_dict[index] += (i + 1) / (len(indices) + 1)

    # 根据评分机制选取前5个索引
    top5_indices = sorted(score_dict.items(), key=lambda item: (-item[1], item[0]))[:5]

    if len(top5_indices) < 5:
        # 对整个文本进行搜索，并将结果补足到top5中
        additional_search_results = search_by_word(my_test_data, model, _index, my_caption, k=5)  # word传入整个文本,指定k=5
        additional_indices = set(additional_search_results) - set([index for index, _ in top5_indices])
        additional_indices = sorted(additional_indices, key=lambda x: additional_search_results.index(x))[
                             :5 - len(top5_indices)]
        top5_indices.extend(additional_indices)

    return [index for index, _ in top5_indices]



st.title("模型测试页面")

dataset_name = st.sidebar.selectbox("选择数据集", [None,"泰迪杯2024B", "flicker30k"])
if not dataset_name:
    st.stop()

# 获取pts文件夹下所有文件名称
pt_list = glob.glob("./pts/*.ckpt")
pt_path = st.selectbox("选择模型参数", pt_list)
# 加载模型
# 利用session管理model
if "pt_path" not in st.session_state:
    st.session_state["pt_path"] = pt_path

if pt_path != st.session_state.get("pt_path", None) or "model" not in st.session_state:
    model = my_model(pt_path)
    if torch.cuda.is_available():
        model = model.to('cuda')
    st.session_state["model"] = model
    st.session_state["pt_path"] = pt_path

model = st.session_state["model"]
# 获取数据
mode = st.sidebar.selectbox("选择模式", ["仅测试", "more"])
if mode == "仅测试":
    json_list = ["test_data.json", ]
else:
    json_list = [
        "",
        "test_data.json",
        "train_data.json",
        "val_data.json"
    ]
json_path = st.selectbox("选择数据", json_list)
if json_path != "":
    my_test_data, df = get_dataset_data("./jsons/"+dataset_name+"/"+json_path, dataset_name=dataset_name)

# 加载全部数据
if st.sidebar.button("数据集展示"):
    st.dataframe(df.head())

captions, img_paths = [], []
for i, caption in enumerate(my_test_data["CAPTIONS"]):
    captions.append(caption)

for i, img_path in enumerate(my_test_data["IMAGES"]):
    img_paths.append(img_path)

# ----------------------------------文搜图------------------------------------------------------
st.sidebar.markdown("## 文搜图")
# 文搜图，保存图向量到I_index
I_index = faiss.IndexFlatIP(1024)  #等价于 faiss.IndexFlatIP(d)   :d维度，  index_type索引类型，  metric_type度量参数
I_index_map = faiss.IndexIDMap(I_index)

# print("创建成功")
# print(I_index.is_trained) # >>> True
# 分批保存图向量
# batch_size = st.sidebar.selectbox("分批保存batch_size", [5, 10, 20, 30])
# cut = range(0, len(my_test_data["IMAGES"]), batch_size)
# print(list(cut))
# print("开始分组")
# 正则检测I[].index是否存在
with st.expander("选择index"):
    if not glob.glob("./indexs/I*.index"):
        st.warning("I.index不存在，请先运行get_test_index.py获取I.index与T.index,并自己加上日期命名")
    else:
        st.info("已存在I.index")
        # glob正则匹配I[可选].index
        I_index_path = st.selectbox("选择I.index版本（日期为版本）",glob.glob("./indexs/I*.index"))
        I_index_map = faiss.read_index(I_index_path)


    if not glob.glob("./indexs/T*.index"):
        st.warning("T.index不存在，请先运行get_test_index.py获取I.index与T.index,并自己加上日期命名")
    else:
        st.info("已存在T.index")
        # glob正则匹配I[可选].index
        T_index_path = st.selectbox("选择T.index版本（日期为版本）", glob.glob("./indexs/T*.index"))
        T_index_map = faiss.read_index(T_index_path)
# 召回率之后不这么计算
# k = st.sidebar.selectbox("选择k", [1, 5, 10,20])
# if st.sidebar.button(f"全部处理，计算Recall@{k}"):
#     count = 0
#     mybar = st.progress(0,"处理进度条")
#     for i in range(len(captions)):
#         # print(i,captions[i])
#         # print(img_paths[0:1],captions[0])
#         # print(os.getcwd())
#         _, text_code = get_features(model=model, img_paths=img_paths[0],
#                                     captions=captions[i])
#         text_code = text_code.cpu()
#         # print(text_code.shape)
#
#         D,I = I_index_map.search(text_code,k)
#         # print(i,captions[i],I[0])
#
#         if i in I[0]:
#             count += 1
#         mybar.progress((i+1)/len(captions),text=f"当前完成{(i+1)/len(captions)*100}%")
#     print(f"Recall@{k}=",count/len(captions))
#     st.write(f"Recall@{k}=",count/len(captions))
#     st.info("计算完成")
#     with open("Recall.txt","a+") as f:
#         # 记录时间和R@k
#         f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())},Recall@{k}={count/len(captions)}")
#         f.write("\n")

if glob.glob("./indexs/I*.index"):
    # 数据编号查找
    my_num = st.sidebar.text_input("输入测试数据编号(priority:0)", "")
    my_num = int(my_num) if not my_num == "" else ""
    if my_num != "":
        if my_num<0 or my_num>=len(captions):
            st.sidebar.error(f"输入编号有误,请确保在0-{len(captions)-1}之间")
            st.stop()

    T2I_mode = st.sidebar.select_slider("选择文搜图模式",["直接搜索","自定义文本查找","分词后名词搜图"],value="直接搜索")
    print(T2I_mode)
    isModified = True if T2I_mode == "自定义文本查找" else False
    isSplited = True if T2I_mode == "分词后名词搜图" else False
    if isModified:
        question = st.text_input("输入你对于文本的处理","主要内容是什么?请使用原文单词")

    text_empty = st.empty()
    result_empty = st.empty()
    if isModified:
        my_text = st.sidebar.text_input("自定义测试数据priority:1", "")

    if st.sidebar.button("点击搜索"):
        # 直接搜索
        if my_num != "" and not isModified and not isSplited:
            text_empty.markdown("### 测试单个文本搜k=5图片")
            raw_captions = (captions[my_num])
            mycaption = raw_captions if not isModified else get_summary_text(raw_captions,question=question)
            _, text_code = get_features(model=model, img_paths=img_paths[0],
                                        captions=mycaption)
            text_code = text_code.cpu()
            D,I = I_index_map.search(text_code, 5)
            # 分2行3列展示
            col1, col2, col3 = result_empty.columns(3)
            st.write(f"索引的文本为：{raw_captions}")
            if isModified:
                st.write(f"修饰后的文本：{mycaption}")
            for i in range(5):
                col = col1 if i < 2 else col2 if i < 4 else col3
                col.image(img_paths[I[0][i]], caption="i:"+captions[I[0][i]], use_column_width=True)
            st.info("搜索完成")
            st.image(img_paths[my_num],caption="原图")

        # 自定义文本查找
        if isModified:

            if my_text!="":
                if my_num != "":
                    st.warning("请先清空上方数字")
                    st.stop()
                st.sidebar.text(f"修饰后:{get_summary_text(my_text,question=question)},此处不采用修饰后搜索")
                text_empty.markdown("### 自定义文本搜k=5图片")
                _, text_code = get_features(model=model, img_paths=img_paths[0],
                                            captions=my_text)
                text_code = text_code.cpu()
                # D, I = I_index_map.search(text_code, 5)
                D, I = faiss_search(I_index_map, text_code, 5)
                # 分2行3列展示
                col1, col2, col3 = result_empty.columns(3)

                st.write(f"自定义索引的文本为：{my_text}")
                for i in range(5):
                    col = col1 if i < 2 else col2 if i < 4 else col3
                    col.image(img_paths[I[0][i]], caption=captions[I[0][i]], use_column_width=True)
                st.info("搜索完成")

        # 分词后名词搜图
        if isSplited:
            my_caption = my_test_data["CAPTIONS"][my_num]
            words = cut_text(my_caption)
            time1 = time.time()
            top5_id_ls = search_by_split(my_test_data,model,I_index_map,my_num)
            st.write(f"耗时{time.time()-time1}")
            text_empty.markdown("### 分词后名词搜图")
            result_empty.markdown("### 结果")
            st.write(f"原文本：{my_caption}")
            col1, col2, col3 = result_empty.columns(3)
            for i in range(5):
                col = col1 if i < 2 else col2 if i < 4 else col3
                col.image(img_paths[top5_id_ls[i]], caption=captions[top5_id_ls[i]], use_column_width=True)
            st.image(img_paths[my_num], caption="原图："+captions[my_num])
# ----------------------------------------图搜文---------------------------------------------

st.sidebar.markdown("## 图搜文")
my_image_id = st.sidebar.text_input("输入图片编号", "")
my_image_id = int(my_image_id) if not my_image_id == "" else ""
if my_image_id != "":
    if my_image_id<0 or my_image_id>=len(img_paths):
        st.sidebar.error(f"输入编号有误,请确保在0-{len(img_paths)-1}之间")
        st.stop()
image_empty = st.empty()
result_empty = st.empty()
if my_image_id != "":
    image_empty.markdown("### 测试单个图搜k=5文本")
    image_code, _ = get_features(model=model, img_paths=img_paths[my_image_id],
                                 captions="")
    image_code = image_code.cpu()
    # D, I = T_index_map.search(image_code, 5)
    D, I = faiss_search(T_index_map, image_code, 5)
    # 第一列展示原图，以及对应的文本。第二列展示搜索到的文本
    col1, col2 = result_empty.columns(2)
    col1.image(img_paths[my_image_id], caption=f"{i}:"+captions[my_image_id], use_column_width=True)
    st.write(f"索引的图片为：{img_paths[my_image_id]}")
    st.write(f"{img_paths[my_image_id].split('.')}")
    for i in range(5):
        col2.write(captions[I[0][i]])
    st.info("搜索完成")
