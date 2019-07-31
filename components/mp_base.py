import torch.nn as nn
import torch
import torchvision
from torchvision.ops import MultiScaleRoIAlign
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MedPoseBase(nn.Module):

    def __init__(self, num_keypoints, num_rpn_props):
        super(MedPoseBase, self).__init__()
        '''
        base cnn for extracting feature maps (want a base
        model that has been trained for keypoint detection
        because feature maps will be more useful in extracting
        context for pose detection)
        '''
        keypoint_rcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                pretrained=True,
                num_keypoints=num_keypoints,
                rpn_post_nms_top_n_test=num_rpn_props)
        '''
        truncate base cnn model so that instead of regressing to
        keypoints and classification, a feature map from an earlier
        layer is extracted (to do this we extract the transform module
        and the backbone module)
        '''
        self.fx_map_transform = keypoint_rcnn.transform
        self.fx_map_backbone = keypoint_rcnn.backbone
        '''
        the base cnn also has a region proposal network that can be
        extracted out for our purposes
        '''
        self.rpn = keypoint_rcnn.rpn
        '''
        crop and resize feature maps in location indicated by bounding boxes
        of region proposals
        '''
        self.roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3, 4],
                output_size=7,
                sampling_ratio=2)

    def _extract_feature_maps(self, x):
        if len(x) >= 3:
            x = x[-3:]
        images, targets = self.fx_map_transform(x, None)
        return self.fx_map_backbone(images.tensors), images, targets

    def _extract_regions(self, features, images, targets):
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        return self.rpn(images, features, targets)[0]

    def _extract_roi(self, features, images, region_props):
        return self.roi_pool(features, region_props, images.image_sizes)

    def extract_base_features(self, x):
        feature_maps, images, targets = self._extract_feature_maps(x)
        region_props = self._extract_regions(feature_maps, images, targets)
        # visualize a region proposal
        #print(len(region_props), region_props[0].shape)
        #print("image dimensions", x[0].shape)
        #print("one region proposal", torch.round(region_props[0][0]))
        #fig, ax = plt.subplots(1)
        #ax.imshow(x[0].permute(1, 2, 0).cpu().numpy())
        #region_props_rounded = torch.round(region_props[0])
        #x1 = region_props_rounded[0][0].item()
        #y1 = region_props_rounded[0][1].item()
        #x2 = region_props_rounded[0][2].item()
        #y2 = region_props_rounded[0][3].item()
        #print(x1, y1, x2, y2)
        #rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), linewidth=1, edgecolor='r', facecolor='none')
        #ax.add_patch(rect)
        #plt.show()
        region_features = self._extract_roi(feature_maps, images, region_props)
        return feature_maps, region_features
