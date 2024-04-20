from torch import nn
import torch


# 损失函数
class TripleNetLoss(nn.Module):
    def __init__(self, margin=0.2, hard_negative=False):
        super(TripleNetLoss, self).__init__()
        self.margin = margin
        self.hard_negative = hard_negative

    def forward(self, ie, te):
        """
        参数：
            ie: 图像表示，为 VSEPP 返回的 image_code
            te: 文字表示，为 VSEPP 返回的 text_code
        """
        scores = ie.mm(te.t())

        diagonal = scores.diag().view(ie.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # 图像为锚
        cost_i = (self.margin + scores - d1).clamp(min=0)
        # 文本为锚
        cost_t = (self.margin + scores - d2).clamp(min=0)

        # 损失矩阵对角线上的值不参与运算
        mask = torch.eye(scores.size(0), dtype=torch.bool)
        I = torch.autograd.Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_i = cost_i.masked_fill_(I, 0)
        cost_t = cost_t.masked_fill_(I, 0)

        # 寻求困难样本
        if self.hard_negative:
            cost_i = cost_i.max(1)[0]
            cost_t = cost_t.max(0)[0]

        return cost_t.sum() + cost_i.sum()
