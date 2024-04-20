import glob
import os.path
import time
from typing import Union

import faiss
from PIL import Image
import pandas as pd
import streamlit as st

import json

from torch import cosine_similarity
from transformers import AutoImageProcessor
from get_test_index import preprocess_I,preprocess_T,get_features
from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
from models import mktrainval, config
import torch


# 缓存model
@st.cache_data
def my_model(_pt_path="best_model0416.ckpt"):
    model = get_model(pt_path=_pt_path)
    print("model加载完毕")
    return model


@st.cache_data
def get_dataset_data(json_path, all_data_path="./data/ImageWordData.csv"):
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.read_csv(all_data_path, encoding="utf-8")
    return data, df




st.title("模型测试页面")

# 获取pts文件夹下所有文件名称
pt_list = glob.glob("./pts/*.ckpt")
pt_path = st.selectbox("选择模型参数", pt_list)
# 加载模型
if pt_path:
    model = my_model(pt_path)
    if torch.cuda.is_available():
        model = model.to('cuda')
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
    my_test_data, df = get_dataset_data(json_path)

# 加载全部数据
if st.sidebar.button("数据集展示"):
    st.dataframe(df.head())
st.sidebar.markdown("## 文搜图")
captions, img_paths = [], []
for i, caption in enumerate(my_test_data["CAPTIONS"]):
    captions.append(caption)

for i, img_path in enumerate(my_test_data["IMAGES"]):
    img_paths.append(img_path)

# 文搜图，保存图向量到I_index
I_index = faiss.IndexFlatIP(1024)  #等价于 faiss.IndexFlatIP(d)   :d维度，  index_type索引类型，  metric_type度量参数
I_index_map = faiss.IndexIDMap(I_index)

print("创建成功")
# print(I_index.is_trained) # >>> True
# 分批保存图向量
batch_size = st.sidebar.selectbox("分批保存batch_size", [5, 10, 20, 30])
cut = range(0, len(my_test_data["IMAGES"]), batch_size)
# print(list(cut))
print("开始分组")
# 正则检测I[].index是否存在
if not glob.glob("./indexs/I*.index"):
    st.warning("I.index不存在，请先运行get_test_index.py获取I.index与T.index,并自己加上日期命名")
else:
    st.info("已存在I.index")
    # glob正则匹配I[可选].index
    I_index_path = st.selectbox("选择I.index版本（日期为版本）",glob.glob("./indexs/I*.index"))
    I_index_map = faiss.read_index(I_index_path)

k = st.sidebar.selectbox("选择k", [1, 5, 10,20])
if st.sidebar.button(f"全部处理，计算Recall@{k}"):
    count = 0
    mybar = st.progress(0,"处理进度条")
    for i in range(len(captions)):
        # print(i,captions[i])
        # print(img_paths[0:1],captions[0])
        # print(os.getcwd())
        _, text_code = get_features(model=model, img_paths=img_paths[0],
                                    captions=captions[i])
        text_code = text_code.cpu()
        # print(text_code.shape)

        D,I = I_index_map.search(text_code,k)
        # print(i,captions[i],I[0])

        if i in I[0]:
            count += 1
        mybar.progress((i+1)/len(captions),text=f"当前完成{(i+1)/len(captions)*100}%")
    print(f"Recall@{k}=",count/len(captions))
    st.write(f"Recall@{k}=",count/len(captions))
    st.info("计算完成")
    with open("Recall.txt","a+") as f:
        # 记录时间和R@k
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())},Recall@{k}={count/len(captions)}")
        f.write("\n")

if glob.glob("./indexs/I*.index"):
    my_num = st.sidebar.text_input("输入测试数据编号(priority:0)", "")
    my_num = int(my_num) if not my_num == "" else ""
    if my_num != "":
        if my_num<0 or my_num>=len(captions):
            st.sidebar.error(f"输入编号有误,请确保在0-{len(captions)-1}之间")
            st.stop()
    text_empty = st.empty()
    result_empty = st.empty()
    if my_num != "":
        text_empty.markdown("### 测试单个文本搜k=5图片")
        _, text_code = get_features(model=model, img_paths=img_paths[0],
                                    captions=captions[my_num])
        text_code = text_code.cpu()
        D,I = I_index_map.search(text_code, 5)
        # 分2行3列展示
        col1, col2, col3 = result_empty.columns(3)
        st.write(f"索引的文本为：{captions[my_num]}")
        for i in range(5):
            col = col1 if i < 2 else col2 if i < 4 else col3
            col.image(img_paths[I[0][i]], caption=captions[I[0][i]], use_column_width=True)
        st.info("搜索完成")
        st.image(img_paths[my_num],caption="原图")
    my_text = st.sidebar.text_input("自定义测试数据priority:1", "")
    if my_text!="":
        if my_num != "":
            st.stop()
        text_empty.markdown("### 自定义文本搜k=5图片")
        _, text_code = get_features(model=model, img_paths=img_paths[0],
                                    captions=my_text)
        text_code = text_code.cpu()
        D, I = I_index_map.search(text_code, 5)
        # 分2行3列展示
        col1, col2, col3 = result_empty.columns(3)
        st.write(f"自定义索引的文本为：{my_text}")
        for i in range(5):
            col = col1 if i < 2 else col2 if i < 4 else col3
            col.image(img_paths[I[0][i]], caption=captions[I[0][i]], use_column_width=True)
        st.info("搜索完成")


st.sidebar.markdown("## 图搜文")
if not glob.glob("./indexs/T*.index"):
    st.warning("T.index不存在，请先运行get_test_index.py获取I.index与T.index,并自己加上日期命名")
else:
    st.info("已存在T.index")
    # glob正则匹配I[可选].index
    T_index_path = st.selectbox("选择T.index版本（日期为版本）",glob.glob("./indexs/T*.index"))
    T_index_map = faiss.read_index(T_index_path)

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
    D, I = T_index_map.search(image_code, 5)
    # 第一列展示原图，以及对应的文本。第二列展示搜索到的文本
    col1, col2 = result_empty.columns(2)
    col1.image(img_paths[my_image_id], caption=captions[my_image_id], use_column_width=True)
    st.write(f"索引的图片为：{img_paths[my_image_id]}")
    for i in range(5):
        col2.write(captions[I[0][i]])
    st.info("搜索完成")
