import torch
from torch import nn
import copy
import torch.nn.functional as F

class ImageCNN(nn.Module):
    def __init__(self, num_outputs, in_channels, out_channels, conv_kernel, pool_kernel):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, pool_kernel),
            nn.ReLU()
        )
        self.linear = nn.Linear(16*6*6, num_outputs)

    def forward(self, sent):
        #sent_emb = self.embeddings(sent)
        sent_emb = sent
        b = sent_emb.size()[0]
        sent_hidden = self.conv1(sent_emb)
        # sent_hidden = F.max_pool2d(sent_hidden, (sent_hidden.size(2), sent_hidden.size(3))).squeeze()
        sent_hidden = F.max_pool2d(sent_hidden, (2, 2))
        sent_hidden = self.conv2(sent_hidden)
        sent_hidden = F.max_pool2d(sent_hidden, (2, 2))
        #print(sent_hidden.size())
        logits = self.linear(sent_hidden.reshape(b, -1))

        return logits


if __name__ == '__main__':
    model = Model_NP()