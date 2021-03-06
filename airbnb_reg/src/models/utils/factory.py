import logging

logger = logging.getLogger(__name__)

from src.models.models import MTResnetAggregate
from ..utils.global_avg_pooling import GlobalAvgPool2dResNext
from torch.nn import Linear


def to_sdl(model, args):
    # Add global_avg_pool and embedding matrix
    num_features = model.num_features
    model = model.body
    model.add_module('global_avg_pool', GlobalAvgPool2dResNext())
    model.add_module('embedding', Linear(num_features, args.num_rows * args.wordvec_dim, bias=False).cuda())
    return model


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    # Make sure no bottlneck_head is used
    model_params['args'].do_bottleneck_head = False
    model_params['args'].bottleneck_features = None

    if args.model_name == 'mtresnetaggregate':
        model = MTResnetAggregate(model_params)
    # if args.model_name == 'tresnet_m':
    #     model = TResnetM(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    # if not args.pretrain_backbone:
    #     model = to_sdl(model, args)

    return model
