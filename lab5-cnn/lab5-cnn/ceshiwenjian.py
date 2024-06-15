# import torch
#
# def corr2d(X, K):
#     batch_size, channels, height, width = X.shape
#     out_channels, in_channels, kh, kw = K.shape
#     Y = torch.zeros(batch_size, out_channels, height - kh + 1, width - kw + 1)
#     for i in range(Y.shape[2]):
#         for j in range(Y.shape[3]):
#             Y[:, :, i, j] = torch.sum(X[:, :, i:i + kh, j:j + kw] * K, dim=(1, 2))
#     return Y
#
# # 示例使用
# X = torch.randn(64, 3, 25, 25)  # 保持通道数为3
# K = torch.randn(16, 3, 8, 8)     # 修改卷积核通道数为3
#
# Y = corr2d(X, K)
# print(Y.shape)

import torch


def corr2d(X, K):
    '''
    X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
    K --> (O, I, h, w) where O = out_channel, I = in_channel, h = height of kernel, w = width of kernel
    你需要实现一个Stride为1，Padding为0的窄卷积操作
    Y的大小应为(B, O, H-h+1, W-w+1)
    '''

    B, I, H, W = X.shape
    O, _, h, w = K.shape
    Y = torch.zeros((B, O, H - h + 1, W - w + 1))
    # 将输入张量和卷积核张量展开为二维矩阵，以便进行矩阵乘法
    X_col = torch.nn.functional.unfold(X, kernel_size=(h, w))
    K_col = K.view(O, -1)

    # 执行矩阵乘法
    Y_col = K_col @ X_col

    # 将结果重新整形为输出张量的形状
    Y = Y_col.view(B, O, H - h + 1, W - w + 1)
    #     for b in range(B):
    #         for o in range(O):
    #             for i in range(I):
    #                 for j in range(H - h + 1):
    #                     for k in range(W - w + 1):
    #                         Y[b, o, j, k] += torch.sum(X[b, i, j:j+h, k:k+w] * K[o, i, :, :])

    return Y


# 示例使用
X = torch.randn(2, 3, 5, 5)  # 示例输入张量大小为 (2, 3, 5, 5)
K = torch.randn(4, 3, 3, 3)  # 示例卷积核大小为 (4, 3, 3, 3)

# 调用 corr2d 函数计算卷积操作
Y = corr2d(X, K)
print(Y.shape)  # 打印输出张量的形状

# import torch
#
#
# def corr2d(X,K):
#     """计算二维卷积的操作,步长默认为1"""
#     h,w = K.shape
#     Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             Y[i][j] = (X[i:i+h, j:j+w] * K).sum()
#     return Y
#
# X = torch.arange(9,dtype=torch.float32).reshape((3,3))
# K = torch.arange(4,dtype=torch.float32).reshape((2,2))
#
# print(corr2d(X,K))
