from argparse import Namespace

from transformers import AutoTokenizer

DATEINFO="0416"
MY_INFO = """
本次训练，于4/16开始，修改了图像特征处理模型的归一化问题，调整batch_size为32
"""
# 模型参数
config = Namespace(
      captions_per_image=1,
      batch_size=32,
      embed_size=1024,
      image_model="vit-base-patch16-224-in21k",
      text_model="bge-large-zh-v1.5",
      tokenizer=AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5'),
      finetuned=True,
      learning_rate=0.00002,
      lr_update=15,       # 学习率更新的epoch， 15个epoch后学习率变为原来的
      min_learning_rate=0.000002,
      margin=0.2,         # 损失函数相关参数，用于比较a,p,n之间的距离,a是anchor,p是positive,n是negative
      hard_nagative=True, # 损失函数相关参数，使用了困难负样本
      num_epochs=45,
      grad_clip=2,
      evaluate_step=250,       # 每隔多少步在验证集上测试一次
      checkpoint=None,        # 如果不为 None，则利用该变量路径中的模型继续训练
      best_checkpoint="./model/best_model"+DATEINFO+".ckpt",   # 表现最佳的模型的保存路径
      last_checkpoint="./model/last_model"+DATEINFO+".ckpt"    # 训练完成时的模型的保存路径
  )