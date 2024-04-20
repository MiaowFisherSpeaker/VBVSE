import torch
# 优化器
def get_optimizer(model, config):
    params = filter(lambda p : p.requires_grad, model.parameters())
    return torch.optim.Adam(params=params, lr=config.learning_rate)


# 学习率变化
def adjust_learning_rate(optimizer, epoch, config):
    """
    每隔 lr_updata 个轮次，学习率减少至当前的二分之一
    参数：
        optimizer: 优化器
        epoch: 训练轮次
        config: 模型超参数和辅助变量
    """
    lr = config.learning_rate *(0.5 ** (epoch// config.lr_update))
    lr = max(lr, config.min_learning_rate)
    for param_group in optimizer.param_groups:  # 参考： https://blog.csdn.net/weixin_45464524/article/details/130456843
        param_group['lr']= lr