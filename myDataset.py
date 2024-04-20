# 处理数据集

import json
import os
import numpy as np
import pandas as pd
from PIL import Image

from collections import defaultdict


def create_dataset(image_folder="./data/ImageData/", output_folder="./"):
    """
    参数：
        dataset:数据集名称
        captions_per_image:每张图片对应的文本描述数
        # min_word_count:仅考虑在数据集中（除测试集外）出现5次的词
        # max_len:文本描述包含的最大单词数。如果文本描述超过该值则截断
    输出：
        一个词典文件：vocab.json
        三个数据集文件：train_data.json val_data.json test_data.json
    """

    # karpathy_json_path = "./dataset_flickr8k.json"  # 读取json文件
    if not image_folder:
        image_folder = "./data/ImageData/"  # 图片文件夹
    if not output_folder:
        output_folder = "./"  # 保存处理结果的文件夹

    # 读取数据集文本描述的json文件
    # with open(file=karpathy_json_path, mode="r") as j:
    #     data = json.load(fp=j)

    # 改为读取csv文件
    data = pd.read_csv("./data/ImageWordData.csv")

    image_paths = defaultdict(
        list)  # collections.defaultdict() 参考：https://zhuanlan.zhihu.com/p/345741967 ; https://blog.csdn.net/sinat_38682860/article/details/112878842
    image_captions = defaultdict(list)

    np.random.seed(64)

    new_image_paths, new_image_captions = [], []  # 存储处理异常后的数据，合并为新data
    # 先处理异常
    for i, row in data.iterrows():  # 读取每张图片
        captions = row["caption"]
        img_path = row["image_id"]

        path = os.path.join(image_folder,
                            img_path)  # 读取图片路径:"./images/img['filename']" 这里img['filename']为图片名字 os.path.join()函数用于路径拼接文件路径，可以传入多个路径 参考：https://blog.csdn.net/swan777/article/details/89040802

        img = Image.open(path)
        # 检测异常
        if img.size[1] != 1:
            new_image_paths.append(path)  # 保存每张图片路径
            new_image_captions.append(captions)  # 保存每张图片对应描述文本

        else:  # 对异常数据的处理
            print(f"捕获异常数据位于:{path}")
            continue  # 这行虽然没用，反正意思就是捕获异常也继续保存
    data = pd.DataFrame()
    data["image_id"] = new_image_paths
    data["caption"] = new_image_captions

    # 划分数据集

    # 初始设定全部为 'val'
    data['split'] = 'val'
    # train,val,test = 7:2:1 
    # 计算70%训练数据和20%验证数据的数量
    train_num = int(len(data) * 0.7)
    val_num = int(len(data) * 0.2)

    # 使用numpy的random.choice函数，在所有的行index中随机选择70%的行，将'split'列的值设为"train"
    train_indices = np.random.choice(data.index, size=train_num, replace=False)
    data.loc[train_indices, 'split'] = 'train'

    # 移除已被选择作为训练数据的行index
    rest_indices = data.index.difference(train_indices)

    # 在剩余的数据中随机选择20%作为验证数据
    val_indices = np.random.choice(rest_indices, size=val_num, replace=False)
    data.loc[val_indices, 'split'] = 'val'

    # 把剩余30%的数据中的'split'列的值设为"test"
    test_indices = rest_indices.difference(val_indices)
    data.loc[test_indices, 'split'] = 'test'

    print(data.groupby(by='split').count())

    # """
    # 执行完以上步骤后得到了：vocab, image_captions, image_paths
    # vocab 为一个字典结构，key为各个出现的词; value为这个词出现的个数
    # image_captions 为一个字典结构，key为"train","val"; val为列表，表中元素为一个个文本描述的列表
    # image_paths 为一个字典结构，key为"train","val"; val为列表，表中元素为图片路径的字符串

    for i, row in data.iterrows():
        split = row["split"]
        captions = row["caption"]
        img_path = row["image_id"]
        image_paths[split].append(img_path)
        image_captions[split].append(captions)

        # 可运行以下代码验证：
    print(image_paths["train"][1])
    print(image_captions["train"][1])

    # 整理数据集
    for split in image_paths:  # 只会循环三次 split = "train" 、 split = "val" 和 split = "test"

        imgpaths = image_paths[split]  # type(imgpaths)=list
        imcaps = image_captions[split]  # type(imcaps)=list
        # enc_captions = []

        # for i, path in enumerate(imgpaths):

        #     # 合法性检测，检查图像时候可以被解析
        #     img = Image.open(path)                  # 参考：https://blog.csdn.net/weixin_43723625/article/details/108158375

        #     # 如果图像对应的描述数量不足，则补足
        #     if len(imcaps[i]) < captions_per_image:
        #         filled_num = captions_per_image - len(imcaps[i])
        #         captions = imcaps[i]+ [random.choice(imcaps[i]) for _ in range(0, filled_num)]
        #     else:
        #         captions = random.sample(imcaps[i],k=captions_per_image)        # 打乱文本描述 参考：https://blog.csdn.net/qq_37281522/article/details/85032470

        #     assert len(captions)==captions_per_image

        #     # for j,c in enumerate(captions):
        #     #     # 对文本描述进行编码
        #     #     enc_c = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in c] + [vocab["<end>"]]
        #     #     enc_captions.append(enc_c)
        #     # 这里先采用原生处理，不编码了

        # # assert len(imgpaths)* captions_per_image == len(enc_captions)
        # assert len(imgpaths)* captions_per_image == len(imcaps)

        # data = {"IMAGES" : imgpaths,
        #         "CAPTIONS" : enc_captions}
        data = {"IMAGES": imgpaths,
                "CAPTIONS": imcaps}

        # 储存训练集，验证集，测试集
        with open(os.path.join(output_folder, split + "_data.json"), 'w') as fw:
            json.dump(data, fw)

if __name__ == '__main__':
    create_dataset()
