import json
import os
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


class ImageTextDataset(Dataset):
    """
      Pytorch 数据类，用于 Pytorch Dataloader 来按批次产生数据
    """

    def __init__(self, dataset_path, split, captions_per_image=1, transform=None):
        """
        参数：
            dataset_path: json 格式数据文件路径
            # vocab_path: json 格式字典文件路径
            split: "tarin", "val", "test"
            captions_per_image: 每张图片对应的文本描述数
            # max_len: 文本描述最大单词量
            transform: 图像预处理方法
        """
        self.split = split
        assert self.split in {"train", "val",
                              "test"}  # assert的应用 参考：https://blog.csdn.net/TeFuirnever/article/details/88883859
        self.cpi = captions_per_image
        # self.max_len = max_len

        # 载入图像
        with open(dataset_path, "r") as f:
            self.data = json.load(f)

        # # 载入词典
        # with open(vocab_path, "r") as f:
        #     self.vocab = json.load(f)

        # 图像预处理流程
        self.transform = transform

        # 数据量
        self.dataset_size = len(self.data["CAPTIONS"])

    def __getitem__(self, i):
        """
        参数：
            i: 第 i 张图片
        """
        # 第 i 个样本描述对应第 (i // captions_per_image) 张图片
        img = Image.open(self.data['IMAGES'][i // self.cpi]).convert(
            'RGB')  # 参考：https://blog.csdn.net/nienelong3319/article/details/105458742
        # 不对图像再处理，相信官方的函数
        # if img.mode != 'RGB': #粗暴一点
        #   img = img.convert('RGB')
        # 如果有图像预处理流程，进行图像预处理
        if self.transform is not None:
            img = self.transform(img)

        caplen = len(self.data["CAPTIONS"][i])
        # pad_caps = [self.vocab['<pad>']] * (self.max_len + 2 - caplen)
        # caption = torch.LongTensor(self.data["CAPTIONS"][i] + pad_caps)         # 类型转换 参考：https://blog.csdn.net/qq_45138078/article/details/131557441
        # 这里不进行长度统一，因为所用模型输出为1024,但是不得不转换

        # text_model = TextRepExtractor(1024,"bge-large-zh-v1.5") # 这里打算直接用模型,对应得修改VSE++前向
        # caption = torch.LongTensor(self.data["CAPTIONS"][i]) #这个需要做词汇表映射
        # 这里采用tokenizer的方式
        caption = self.data["CAPTIONS"][i]
        # 这里思来想去还是做不了处理
        # caption = text_model(caption)

        return img, caption, caplen

    def __len__(self):
        return self.dataset_size


def mktrainval(data_dir, batch_size, workers=8):
    """
    参数：
        data_dir: json 文件夹位置
        # vocab_path: 词典位置
        batch_size: batch大小
        worker: 运行进程数 default = multiprocessing.cpu_count()
    """
    # 参考：https://zhuanlan.zhihu.com/p/476220305
    # train_tx = transforms.Compose([
    #     transforms.Resize(256),                                                         # 缩放
    #     transforms.RandomCrop(224),                                                     # 随机裁剪
    #     transforms.ToTensor(),                                                          # 用于对载入的图片数据进行类型转换，将图片数据转换成Tensor数据类型的变量
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # 标准化，这里的均值和方差为在ImageNet数据集上抽样计算出来的
    # ])

    # val_tx = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),                                                     # 中心裁剪
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')  # 处理得到的是字典，取pixel_values

    # Tx = lambda x:processor(x)["pixel_values"] # 对VSE++前向传播修改后不需要,但为了dataloader输出维度统一，还是加上[...]
    def Tx(x):
        # Txx = processor(x,return_tensors="pt")
        # if len(Txx) > 1:
        #   print("出大事了")
        # try:
        #   print(f"你传入的x合法是{x}")
        #   return Txx["pixel_values"]
        # except:
        #   raise f"你传入的x是_{x}不合法"
        try:
            # 使用 processor 对样本 x 进行处理
            # print(type(x))
            Txx = processor(x, return_tensors="pt")

            # 确保 processor 返回的是一个字典，并且包含 'pixel_values' 键
            if "pixel_values" not in Txx:
                raise KeyError("The processor did not return 'pixel_values'")

            # 返回处理后的 pixel_values
            return Txx["pixel_values"]

        except Exception as e:
            # 如果处理过程中出现任何异常，抛出一个错误消息
            raise Exception(f"Error processing x: {x}. Error: {e}")

    train_set = ImageTextDataset(dataset_path=os.path.join(data_dir, "train_data.json"), split="train", transform=Tx)
    val_set = ImageTextDataset(dataset_path=os.path.join(data_dir, "val_data.json"), split="val", transform=Tx)
    test_set = ImageTextDataset(dataset_path=os.path.join(data_dir, "test_data.json"), split="test", transform=Tx)

    # 构建 pytorch 的 Dataloader 类
    # 参考：https://blog.csdn.net/rocketeerLi/article/details/90523649 ; https://blog.csdn.net/zfhsfdhdfajhsr/article/details/116836851
    train_loder = data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    # 验证集和测试集不需要打乱数据顺序：shuffer = False
    # 参考：https://blog.csdn.net/qq_42940160/article/details/123894759
    val_loder = data.DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False
    )

    test_loder = data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False
    )

    return train_loder, val_loder, test_loder
