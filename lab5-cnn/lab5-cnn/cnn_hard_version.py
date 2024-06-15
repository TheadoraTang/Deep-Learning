# import torch
# from torch import nn
#
# def corr2d(X, K):
#     '''
#     X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
#     K --> (O, I, h, w) where O = out_channel, I = in_channel, h = height of kernel, w = width of kernel
#     你需要实现一个Stride为1，Padding为0的窄卷积操作
#     Y的大小应为(B, O, H-h+1, W-w+1)
#     '''
#
#     # =============
#     # todo: 请根据以上提示补全代码
#     # =============
#     return Y
#
# class Conv2D(nn.Module):
#     def __init__(self, out_channels, in_channels, kernel_size):
#         super(Conv2D, self).__init__()
#         self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size[0], kernel_size[1])))
#         self.bias = nn.Parameter(torch.randn((out_channels)))
#
#     def forward(self, X):
#         Y = corr2d(X, self.weight) + self.bias.view(1, -1, 1, 1)
#         return Y
#
# class MaxPool2D(nn.Module):
#     def __init__(self, pool_size):
#         super(MaxPool2D, self).__init__()
#         self.pool_size = pool_size
#
#     def forward(self, X):
#         '''
#         X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
#         K --> (h, w) where h = height of kernel, w = width of kernel
#         你需要利用以上pool_size实现一个最大汇聚层的前向传播，汇聚层的子区域间无覆盖
#         Y的大小应为(B, I, H/h, W/w)
#         '''
#
#         # =============
#         # todo: 请根据以上提示补全代码
#         # =============
#         return Y
#
# class ImageCNN(nn.Module):
#     def __init__(self, num_outputs, in_channels, out_channels, conv_kernel, pool_kernel):
#         super(ImageCNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             Conv2D(out_channels, in_channels, conv_kernel),
#             nn.ReLU()
#         )
#         self.pool1 = MaxPool2D(pool_kernel)
#         self.linear = nn.Linear(16*5*5, num_outputs)
#
#     def forward(self, feature_map):
#         b = feature_map.size()[0]
#         feature_map = self.conv1(feature_map)
#         feature_map = self.pool1(feature_map)
#         outputs = self.linear(feature_map.reshape(b, -1))
#         return outputs
#
#
# # if __name__ == '__main__':
# #     model = Model_NP()

import torch
from torch import nn


def corr2d(X, K):
    '''
    X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
    K --> (O, I, h, w) where O = out_channel, I = in_channel, h = height of kernel, w = width of kernel
    你需要实现一个Stride为1，Padding为0的窄卷积操作
    Y的大小应为(B, O, H-h+1, W-w+1)
    '''

    B, I, H, W = X.shape
    O, _, h, w = K.shape
    # Y = torch.zeros((B, O, H - h + 1, W - w + 1))

    # 将输入张量和卷积核张量展开为二维矩阵，以便进行矩阵乘法
    X_col = torch.nn.functional.unfold(X, kernel_size=(h, w))
    K_col = K.view(O, -1)

    # 执行矩阵乘法
    Y_col = K_col @ X_col

    # 将结果重新整形为输出张量的形状
    Y = Y_col.view(B, O, H - h + 1, W - w + 1)

    return Y


class Conv2D(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size[0], kernel_size[1])))
        self.bias = nn.Parameter(torch.randn((out_channels)))

    def forward(self, X):
        Y = corr2d(X, self.weight) + self.bias.view(1, -1, 1, 1)
        return Y


class MaxPool2D(nn.Module):
    def __init__(self, pool_size):
        super(MaxPool2D, self).__init__()
        self.pool_size = pool_size

    def forward(self, X):
        '''
        X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
        K --> (h, w) where h = height of kernel, w = width of kernel
        你需要利用以上pool_size实现一个最大汇聚层的前向传播，汇聚层的子区域间无覆盖
        Y的大小应为(B, I, H/h, W/w)
        '''
        B, I, H, W = X.size()  # 提取批次大小、输入通道数、特征图的高度和宽度
        h, w = self.pool_size  # 提取池化大小
        new_H = H // h  # 池化后的新高度
        new_W = W // w  # 池化后的新宽度

        # 将输入张量重塑为一个形状为(B, I, new_H, h, new_W, w)的张量
        X_reshaped = X.view(B, I, new_H, h, new_W, w)
        # 在最后两个维度上取最大值，得到池化结果
        Y, _ = X_reshaped.max(dim=3, keepdim=True)
        Y, _ = Y.max(dim=5, keepdim=True)

        # 压缩维度，得到最终的池化结果
        Y = Y.squeeze(3).squeeze(4)

        return Y

class ImageCNN(nn.Module):
    def __init__(self, input_size, num_outputs, in_channels, out_channels, conv_kernel, pool_kernel):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2D(out_channels, in_channels, conv_kernel),
            nn.ReLU()
        )
        self.pool1 = MaxPool2D(pool_kernel)
        self.linear = nn.Linear(16 * 5 * 5, num_outputs)

    def forward(self, feature_map):
        b = feature_map.size()[0]
        feature_map = self.conv1(feature_map)
        feature_map = self.pool1(feature_map)
        outputs = self.linear(feature_map.reshape(b, -1))
        return outputs
