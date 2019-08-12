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
        self.num_rpn_props = num_rpn_props
        '''
        crop and resize feature maps in location indicated by bounding boxes
        of region proposals
        '''
        self.roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3, 4],
                output_size=7,
                sampling_ratio=2)

    def _extract_feature_maps(self, x):
        images, targets = self.fx_map_transform(x, None)
        return self.fx_map_backbone(images.tensors), images, targets

    def _extract_regions(self, features, images, targets):
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        # print("SANITY CHECK")
        # for fidx in features:
        #     print(features[fidx].get_device())
        # print(images.tensors.get_device(), "IMAGE TENSORS")
        #print(images.tensors.get_device(), [features[i].get_device() for i in features], "checking this")
        return self.rpn(images, features, targets)[0]

    def _extract_roi(self, features, images, region_props):
        return self.roi_pool(features, region_props, images.image_sizes)

    def extract_base_features(self, x):
        batch_size = len(x)
        #print(len(x), len(x[0]), x[0][0].shape)
        base_in = list(map(list, zip(*x)))
        #print(len(base_in), len(base_in[0]), base_in[0][0].shape)
        #exit()
        frame_feature_maps = []
        for frame_batch in base_in:
            feature_maps, images, targets = self._extract_feature_maps(frame_batch)
            '''
            for now, we only operate on one scale feature maps
            TODO: test the efficacy of different scale feature maps
            on improving attention mechanism
            '''
            frame_feature_maps.append(feature_maps[0])
        '''
        we only care about region features from the last iteration
        (only current frame region features used as query)
        '''
        region_props = self._extract_regions(feature_maps, images, targets)
        '''
        visualize region props (default: no)
        '''
        #self._visualize_region_props(images, region_props)
        cf_region_features = self._extract_roi(feature_maps, images, region_props)
        cf_region_features = cf_region_features.view(batch_size, self.num_rpn_props, 
                cf_region_features.shape[1], cf_region_features.shape[2], cf_region_features.shape[3])
        frame_feature_maps = torch.stack(frame_feature_maps, dim=1)
        return frame_feature_maps, cf_region_features

    def _visualize_region_props(self, images, region_props):
        '''
        plot setup
        '''
        fig, ax = plt.subplots(1)
        images = images.tensors
        for img_idx, region_prop in enumerate(region_props):
            # get image for current region_prop
            image = images[img_idx].permute(1,2,0).cpu().numpy()
            ax.imshow(image)
            for region_idx in range(region_prop.shape[0]):
                bbox = torch.round(region_prop[region_idx])
                x1 = bbox[0].item()
                y1 = bbox[1].item()
                x2 = bbox[2].item()
                y2 = bbox[3].item()
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
        plt.show()
        plt.cla()
