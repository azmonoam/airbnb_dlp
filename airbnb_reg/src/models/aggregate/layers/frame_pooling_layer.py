import torch
import torch.nn as nn


def distL1(v1, v2, th):
    v1[v1 > th] = 1
    v2[v2 > th] = 1
    d = torch.sum(torch.abs(v1 - v2))
    return d


class Aggregate(nn.Module):
  def __init__(self, sampled_frames=None, nvids=None, args=None):
    super(Aggregate, self).__init__()
    self.clip_length = sampled_frames
    self.nvids = nvids
    self.args = args


  def forward(self, x, filenames=None):
    nvids = x.shape[0] // self.clip_length
    x = x.view((-1, self.clip_length) + x.size()[1:])
    o = x.mean(dim=1)
    return o
