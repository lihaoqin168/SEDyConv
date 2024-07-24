import  torch
import numpy as np

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

#2tp/(fp+2tp+fn)
def dice_coef(output, target):#output为预测结果 target为真实结果
    smooth = 1e-5 #防止0除

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
# tp/(fp+tp_fn)
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


#tp/tp+fn
#recall
def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)

#tp/tp+fp
#precision
def ppv(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (output.sum() + smooth)


# self.percentile参数说明
# 表示最大距离分位数，取值范围为0-100，它表示的是计算步骤4中，选取的距离能覆盖距离的百分比，
# 一般是选取了95%，那么在计算步骤4中选取的不是最大距离，而是将距离从大到小排列后，
# 取排名为5%的距离。这么做的目的是为了排除一些离群点所造成的不合理的距离，保持整体数值的稳定性。
# 所以Hausdorff distance也被称为Hausdorff distance-95%。
def hausdorff2(output, target, label_idx=1):
    from mindspore.nn.metrics import HausdorffDistance
    metric = HausdorffDistance(percentile=95)
    metric.clear()
    metric.update(output, target, label_idx)
    distance = metric.eval()
    print(distance)