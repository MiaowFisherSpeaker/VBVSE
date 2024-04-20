from torch import nn
from .embedding_models import ImageRepExtractor, TextRepExtractor


class VSEPP(nn.Module):
    def __init__(self, embed_size, image_model, text_model, finetuned=True):
        """
        参数：
            vocab_size: 词表大小
            word_dim: 词嵌入维度
            embed_size: 对应表示维度，即 RNN 隐藏层维度
            num_layers: RNN隐藏层数
            image_model: 图像表示提取器，ResNet或者VGG19
            finetuned: 是否微调图像表示器
        """
        super(VSEPP, self).__init__()
        self.image_extractor = ImageRepExtractor(embed_size=embed_size, pretrained_model1=image_model,
                                                 finetuned=finetuned)
        self.text_extractor = TextRepExtractor(embed_size=embed_size, pretrained_model2=text_model)

    def forward(self, images, captions, cap_lens):
        # 按照 caption 的长短排序，并对照调整 image 的顺序,这里传入的是类似字典的序列不方便排序
        # sorted_cap_len, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        # print(captions,type(images))
        # images = images[sorted_cap_indices]
        # captions = captions[sorted_cap_indices]
        # cap_lens = sorted_cap_len

        # image_code = self.image_extractor(**images)
        # text_code = self.text_extractor(**captions)
        # # text_code = captions # 这里默认做过处理，不再需要提取特征

        # if not self.training:
        #     # 恢复数据原始输入
        #     _, recover_indices = torch.sort(sorted_cap_indices)
        #     image_code = image_code[recover_indices]
        #     text_code = text_code[recover_indices]

        image_code = self.image_extractor(pixel_values=images)
        text_code = self.text_extractor(**captions)
        return image_code, text_code
