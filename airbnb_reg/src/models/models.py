import torch
import torch.nn as nn
import torchvision

from src.models.aggregate.layers.frame_pooling_layer import Aggregate
from src.models.aggregate.layers.transformer_aggregate import TAggregate
from src.models.resnet.resnet import ResNet
from src.models.tresnet.tresnet import TResNet
from src.models.utils.registry import register_model

__all__ = ['MTResnetAggregate']


class fTResNet(TResNet):

  def __init__(self, aggregate=None, *args, **kwargs):
    super(fTResNet, self).__init__(*args, **kwargs)
    self.aggregate = aggregate
    resnet152_model = torchvision.models.resnet152(pretrained=True)
    model = nn.Sequential(*(list(resnet152_model.children())[:-1]))
    self.body = model


  def forward(self, x, filenames=None, epoch_num=None):
    with torch.no_grad():
        x = self.body(x)
    self.embeddings = self.global_pool(x)

    if self.aggregate:
        if isinstance(self.aggregate,TAggregate):
           self.embeddings, self.attention = self.aggregate(self.embeddings, filenames, epoch_num)
           logits = self.head(self.embeddings)
        else:# CNN aggregation:
            logits = self.head(self.embeddings)
            logits = self.aggregate(nn.functional.softmax(logits, dim=1))
    return logits


class fResNet(ResNet):
  def __init__(self, aggregate=None, **kwargs):
    super().__init__(**kwargs)
    self.aggregate = aggregate

  def forward(self, x):
    with torch.no_grad():
        x = self.body(x)
    if self.aggregate:
      x = self.head.global_pool(x)
      x, attn_weight = self.aggregate(x)
      logits = self.head.fc(self.head.FlattenDropout(x))

    else:
      logits = self.head(x)
    return logits


@register_model
def MTResnetAggregate(model_params):
    """Constructs a medium TResnet model.   Frame Pooling MTResNet (frame pooling)
    """

    in_chans = 3
    num_classes = model_params['num_classes']
    args = model_params['args']
    if 'global_pool' in args and args.global_pool is not None:
        global_pool = args.global_pool
    else:
        global_pool = 'avg'
    do_bottleneck_head = args.do_bottleneck_head
    bottleneck_features = args.bottleneck_features
    remove_model_jit = args.remove_model_jit

    aggregate = None
    if args.use_transformer:
      aggregate = TAggregate(args.album_clip_length, args=args)
    else:
      aggregate = Aggregate(args.album_clip_length, args=args)

    model = fTResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans,
                    # global_pool=global_pool,
                    do_bottleneck_head=do_bottleneck_head,
                    bottleneck_features=bottleneck_features,
                    # remove_model_jit=remove_model_jit,
                    aggregate= aggregate)


    return model