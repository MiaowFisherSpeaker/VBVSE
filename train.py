# 主程序
import os
from tqdm.notebook import tqdm
import logging

import torch
import torch.nn as nn

from models.vsepp import VSEPP
from models.utils.loss import TripleNetLoss
from models.utils.optimizer import get_optimizer, adjust_learning_rate
from models.utils.dataset import mktrainval
from models.utils.evaluate import evaluate
from models import config, MY_INFO

# logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='my_logging.log',
                    filemode='a')
# 版本说明：
logging.info(MY_INFO)

# 是否可以用 cuda
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 启用tokenizer的并行模式
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# 调试
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "./"

train_loader, val_loader, test_loader = mktrainval(data_dir, config.batch_size)

if __name__ == '__main__':
    # 随机初始化或载入已训练模型
    start_epoch = 0
    checkpoint = config.checkpoint
    if checkpoint is None:
        model = VSEPP(
            embed_size=config.embed_size,
            image_model=config.image_model,
            text_model=config.text_model,
            finetuned=config.finetuned
        )
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']

    # 优化器
    optimizer = get_optimizer(model=model, config=config)

    # 模型移动到 GPU， 并进入训练模式
    model.to(device)
    model.train()

    # 损失函数
    loss_fn = TripleNetLoss(margin=config.margin, hard_negative=config.hard_nagative)

    # 训练
    best_res = 0
    print("开始训练")
    logging.info("开始训练")
    for epoch in range(start_epoch, config.num_epochs):

        adjust_learning_rate(optimizer, epoch, config)

        print(f"Epoch:{epoch + 1}/{config.num_epochs}")
        logging.info(f"Epoch:{epoch + 1}/{config.num_epochs}")

        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            caps = list(caps)
            caps = config.tokenizer(caps, padding=True, truncation=False, return_tensors='pt')  #这里对于一批数据需要自动padding
            # print(caps) # dict
            # 数据读取至 GPU
            imgs = imgs.squeeze(1).to(device, non_blocking=True)
            caps = {k: v.to(device, non_blocking=True) for k, v in caps.items()}

            # 向前传播
            image_code, text_code = model(imgs, caps, caplens)

            # print(image_code.shape, text_code.shape)
            # 计算损失
            loss = loss_fn(image_code, text_code)
            loss.backward()

            # 梯度截断
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # 更新参数
            optimizer.step()

            # 当前状态
            state = {
                'epoch': epoch,
                'step': i,
                'model': model,
                'optimizer': optimizer
            }

            if (i + 1) % config.evaluate_step == 0:
                r1_i2t, r1_t2i, _, _ = evaluate(data_loader=val_loader,
                                                model=model,
                                                batch_size=config.batch_size,
                                                captions_per_image=config.captions_per_image
                                                )
                recall_sum = r1_i2t + r1_t2i

                # 选择模型
                if best_res < recall_sum:
                    best_res = recall_sum
                    torch.save(state, config.best_checkpoint)

                torch.save(state, config.last_checkpoint)

                print(f"epoch: {epoch}, step: {i + 1}, loss: {loss.item()}")
                logging.info(f"epoch: {epoch}, step: {i + 1}, loss: {loss.item()}")

    # 用效果最好的模型在测试集上进行测试
    checkpoint = torch.load(config.best_checkpoint)
    model = checkpoint['model']
    r1_i2t, r1_t2i, r5_i2t, r5_t2i = evaluate(data_loader=test_loader, model=model, batch_size=config.batch_size,
                                              captions_per_image=config.captions_per_image)
    print(
        f"Epoch: {checkpoint['epoch']}, \n I2T R@1: {r1_i2t}, T2I R@1: {r1_t2i}, \t I2T R@5: {r5_i2t}, T2I R@5: {r5_t2i}")
    logging.info(
        f"Epoch: {checkpoint['epoch']}, \n I2T R@1: {r1_i2t}, T2I R@1: {r1_t2i}, \t I2T R@5: {r5_i2t}, T2I R@5: {r5_t2i}")
