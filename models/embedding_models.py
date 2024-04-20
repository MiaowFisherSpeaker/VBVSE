import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ImageRepExtractor(nn.Module):
    def __init__(self, embed_size, pretrained_model1='vit-base-patch16-224-in21k', finetuned=True):
        """
        参数:
            embed_size: 维度
            pretrained_model1: 图像表示提取器，如vit-base-patch16-224-in21k
            finetuned: 是否微调参数（默认True）
        """
        super(ImageRepExtractor, self).__init__()

        self.pretrained_model = pretrained_model1
        self.embed_size = embed_size
        self.embed_Linear_layer = nn.Linear(768, self.embed_size)  # vit模型外修改维数768->embed_size

        if self.pretrained_model == "vit-base-patch16-224-in21k":
            net = AutoModel.from_pretrained('google/vit-base-patch16-224-in21k')

            for param in net.parameters():
                param.requires_grad = finetuned
            # 修改输出层
            # net.encoder.layer[-1].output.dense = nn.Linear(net.encoder.layer[-1].output.dense.in_features,embed_size)
            # nn.init.xavier_uniform_(net.encoder.layer[-1].output.dense.weight)      # 参考： https://blog.csdn.net/luoxuexiong/article/details/95772045
            # net.pooler.dense = nn.Linear(net.pooler.dense.in_features,embed_size)
            # nn.init.xavier_uniform_(net.pooler.dense.weight)      # 参考： https://blog.csdn.net/luoxuexiong/article/details/95772045
            # 这里输出层修改不了，不建议修改了

        else:
            raise ValueError("Unknow image model" + self.pretrained_model)
        self.net = net

    def forward(self, **inputs):
        if self.pretrained_model == "vit-base-patch16-224-in21k":
            output = self.net(**inputs).last_hidden_state  # (1,patches=197,dimensions=768)
            # print(output.shape)
            # 这里不过池化层，人工池化
            output = torch.mean(output, dim=1)  # 这里可能有问题
            # print(output.shape)
            output = self.embed_Linear_layer(output)  # 全连接层 (1, embed_size)我们要的向量
            output = F.normalize(output, p=2, dim=1)  # 别忘了归一化

            return output

    def __repr__(self):
        return self.net.__repr__()


class TextRepExtractor(nn.Module):
    def __init__(self, embed_size, pretrained_model2="bge-large-zh-v1.5"):
        """
        参数：
            embed_size: 对应表示维度
            pretrained_model2: 文本表示提取器预训练模型
        """
        super(TextRepExtractor, self).__init__()

        self.embed_dim = embed_size
        self.pretrained_model = pretrained_model2
        if self.pretrained_model == "bge-large-zh-v1.5":
            self.model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')

    def forward(self, **encoded_input):
        """ x输入需要先经过对应的tokenizer """
        if self.pretrained_model == "bge-large-zh-v1.5":
            # out = self.model.encode(x,normalize_embeddings=True) # 归一化，这样计算俩个语句相似度直接点积即可，也为了方便后续处理
            self.model.eval()
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            # 归一化
            out = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            self.model.train()
        return out
