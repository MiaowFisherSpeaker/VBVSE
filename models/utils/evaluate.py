import numpy as np
import torch
from models.config import config


def evaluate(data_loader, model, batch_size, captions_per_image):
    # 切换模型为评估模式
    model.eval()
    image_codes = None
    text_codes = None
    device = next(model.parameters()).device
    N = len(data_loader.dataset)

    for i, (imgs, caps, caplens) in enumerate(data_loader):

        imgs = imgs.squeeze(1).to(device, non_blocking=True)
        caps = list(caps)
        caps1 = caps
        caps = config.tokenizer(caps, padding=True, truncation=False, return_tensors='pt')
        caps = {k: v.to(device, non_blocking=True) for k, v in caps.items()}

        with torch.no_grad():
            image_code, text_code = model(imgs, caps, caplens)
            if image_codes is None:
                # print("呔！")
                # print(caps1)
                image_codes = np.zeros((N, image_code.size(1)))
                text_codes = np.zeros((N, text_code.size(1)))
            # 将图文对应表示存到 numpy 数组中，之后在 CPU 上计算 recall
            st = i * batch_size
            ed = (i + 1) * batch_size

            image_codes[st:ed] = image_code.data.cpu().numpy()
            text_codes[st:ed] = text_code.data.cpu().numpy()
    # 模型切换回训练模式
    model.train()
    return calc_recall(image_codes, text_codes, captions_per_image)


def calc_recall(image_codes, text_codes, captions_per_image):
    # 之所以可以每隔固定数量取图片，是因为前面对图文数据对输入顺序进行了还原
    scores = np.dot(image_codes[::captions_per_image], text_codes.T)
    # 以图检文， 按照从小到大排序
    sorted_scores_indices = (-scores).argsort(axis=1)
    (n_image, n_text) = scores.shape
    ranks_i2t = np.zeros(n_image)
    for i in range(0, n_image):
        # 一张图片对应 cpi 条文本，找到排名最靠前的文本位置
        min_rank = 1e10
        for j in range(i * captions_per_image, (i + 1) * captions_per_image):
            rank = list(sorted_scores_indices[i, :]).index(j)
            if min_rank > rank:
                min_rank = rank
        ranks_i2t[i] = min_rank
    # 以文检图， 按照从小到大排序
    sorted_scores_indices = (-scores).argsort(axis=0)
    ranks_t2i = np.zeros(n_text)
    for i in range(n_text):
        rank = list(sorted_scores_indices[:, i]).index(i // captions_per_image)
        ranks_t2i[i] = rank
    # 最靠前的位置小于 k，即 recall@k， 这里计算了 k 取 1，5，10 时的图文互检的 recall
    r1_i2t = 100.0 * len(np.where(ranks_i2t < 1)[0]) / n_image
    r1_t2i = 100.0 * len(np.where(ranks_t2i < 1)[0]) / n_text
    r5_i2t = 100.0 * len(np.where(ranks_i2t < 5)[0]) / n_image
    r5_t2i = 100.0 * len(np.where(ranks_t2i < 5)[0]) / n_text
    # r10_i2t = 100.0 * len(np.where(ranks_i2t < 10)[0]) / n_image
    # r10_t2i = 100.0 * len(np.where(ranks_t2i < 10)[0]) / n_text

    # return r1_i2t, r1_t2i, r5_i2t, r5_t2i, r10_i2t, r10_t2i
    # return r5_i2t, r5_t2i
    return r1_i2t, r1_t2i, r5_i2t, r5_t2i
