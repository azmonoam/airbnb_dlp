import os
from torch import nn
import torch
from src.utils.utils import trunc_normal_
import pickle
import numpy as np
import pandas as pd

class TransformerEncoderLayerWithWeight(nn.TransformerEncoderLayer):
  def __init__(self, *args, **kwargs):
    super(TransformerEncoderLayerWithWeight, self).__init__(*args, **kwargs)

  def forward(self, src, src_mask=None, src_key_padding_mask=None):
    src2, attn_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask)
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src, attn_weight


class TransformerEncoderWithWeight(nn.TransformerEncoder):
  def __init__(self, *args, **kwargs):
    super(TransformerEncoderWithWeight, self).__init__(*args, **kwargs)

  def forward(self, src, mask=None, src_key_padding_mask=None):
    # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
    r"""Pass the input through the encoder layers in turn.

    Args:
        src: the sequence to the encoder (required).
        mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    output = src

    for mod in self.layers:
      output, attn_weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

    if self.norm is not None:
      output = self.norm(output)

    return output, attn_weight


class TAggregate(nn.Module):
  def __init__(self, clip_length=None, embed_dim=2048, n_layers=2, args=None):
    super(TAggregate, self).__init__()
    self.clip_length = clip_length
    drop_rate = 0.
    self.args = args
    enc_layer = TransformerEncoderLayerWithWeight(d_model=embed_dim, nhead=8)
    self.transformer_enc = TransformerEncoderWithWeight(enc_layer, num_layers=n_layers,
                                                        norm=nn.LayerNorm(
                                                          embed_dim))

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)

    with torch.no_grad():
      trunc_normal_(self.pos_embed, std=.02)
      trunc_normal_(self.cls_token, std=.02)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      with torch.no_grad():
        trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  def forward(self, x, filenames=None, epoch_num=None):
    nvids = x.shape[0] // self.clip_length
    x = x.view((nvids, self.clip_length, -1))
    pre_aggregate = torch.clone(x)
    cls_tokens = self.cls_token.expand(nvids, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    if self.args.transformers_pos:
      x = x + self.pos_embed
    # x = self.pos_drop(x)
    x.transpose_(1, 0)
    o, attn_weight = self.transformer_enc(x)
    o.transpose_(1, 0)
    # save attn_weight as a pickle file
    if filenames and (epoch_num == self.args.epochs):
      att_data = pd.read_csv('{}att_data_{}.csv'.format(self.args.path_output, self.args.start_ts))
      for b in range(nvids):
        # get album name:
        apt_pic_pathes = []
        for fn in range(b * self.clip_length, (b + 1) * self.clip_length):
          apt_pic_pathes.append(os.path.splitext(os.path.basename(filenames[fn]))[0])
        apt_id = filenames[b * self.clip_length].split('/')[-2]
        att_mat = attn_weight[b].cpu().detach().numpy().tolist()
        arg_max = int(torch.argmax(attn_weight[b][0, 1:]))
        meaningful_image_path = apt_pic_pathes[arg_max]
        meaningful_image = os.path.splitext(os.path.basename(meaningful_image_path))[0]
        meaningful_image = int(meaningful_image[meaningful_image.find('_I') + 2:])
        att_data = pd.concat([att_data, pd.DataFrame({'id': apt_id, 'att_mat': [att_mat] , 'pic_order': [apt_pic_pathes],
                            'most_important_pic_path':meaningful_image_path, 'most_important_pic':meaningful_image})],
                             ignore_index=True,        axis=0)
      att_data.to_csv('{}att_data_{}.csv'.format(self.args.path_output, self.args.start_ts),index=False)
    return o[:, 0], attn_weight
