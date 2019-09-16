from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler

from maskrcnn_benchmark.layers import Conv2d

@registry.ROI_KEYPOINT_FEATURE_ENLARGER.register("KeypointRCNNFeatureEnlarger")
class KeypointRCNNFeatureEnlarger(nn.Module):
    def __init__(self, cfg, in_channels):
        super(KeypointRCNNFeatureEnlarger, self).__init__()

        input_features = in_channels
        if cfg.MODEL.MEDPOSE_ON:
            layers = cfg.MODEL.MEDPOSE.CONV_LAYERS
        else:
            layers = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS
        next_feature = input_features
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "conv_fcn{}".format(layer_idx)
            module = nn.Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        return x


def make_roi_keypoint_feature_enlarger(cfg, in_channels):
    func = registry.ROI_KEYPOINT_FEATURE_ENLARGER[
        cfg.MODEL.MEDPOSE.FEATURE_ENLARGER
    ]
    return func(cfg, in_channels)
