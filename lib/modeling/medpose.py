import sys, os
sys.path.append(os.path.abspath(os.path.join('./')))
from core.config import cfg
from components.mp_encoder import MedPoseEncoder
from components.mp_decoder import MedPoseDecoder
from components.mp_layers import MedPoseHistory
from utils.keypoints import keypoints_to_pred_heatmap
import nn as mynn
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time

class med_pose_network(nn.Module):

    def __init__(self, roi_xform_func, spatial_scale):
        super(med_pose_network, self).__init__()
        '''
        initialize roi func and spatial scales from init params
        '''
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        '''
        since feature map has dim size of 256, a conv module is 
        used to increase dim size to model dim size of 1088 (8 * 8 * 17) 
        for later heatmap creation (idea #1)
        '''
        dim_in = cfg.MEDPOSE.MODEL_DIM
        hidden_dim = 1088                   # testing for now that's why its not in config file
        #hidden_dim = 512
        kernel_size = cfg.KRCNN.CONV_HEAD_KERNEL
        pad_size = kernel_size // 2
        module_list = []
        # create module for feature map feature enlargement
        for _ in range(cfg.MEDPOSE.NUM_STACKED_CONVS):
            module_list.append(nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size))
            module_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        self.conv_fcn_fm = nn.Sequential(*module_list)
        # create module for region features feature enlargement
        module_list = []
        dim_in = cfg.MEDPOSE.MODEL_DIM
        for _ in range(cfg.MEDPOSE.NUM_STACKED_CONVS):
            module_list.append(nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size))
            module_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        self.conv_fcn_rf = nn.Sequential(*module_list)
        # create module for keypoint feature preds from region features for better signal propagation
        dim_in = cfg.MEDPOSE.MODEL_DIM
        dim_in = hidden_dim
        self.conv_fcn_kp = nn.Conv2d(dim_in, cfg.MEDPOSE.NUM_KEYPOINTS, 1, 1, padding=0)
        '''
        initialize the MedPose encoder architecture with
        default parameters (encoder attends to local structures
        using feature maps bounded by region proposals as
        queries)
        '''
        encoder_history = [MedPoseHistory() for i in range(cfg.NUM_GPUS)]
        self.encoder = MedPoseEncoder(num_enc_layers=cfg.MEDPOSE.STACK_LAYERS, enc_history=encoder_history, lrnn_window_size=cfg.MEDPOSE.WINDOW_SIZE, roi_map_dim=cfg.MEDPOSE.ROI_MAP_DIM, gpus=range(cfg.NUM_GPUS))
        '''
        fully connected networks for classification (pose detectable
        or not) and keypoint output (location of keypoints)
        '''
        self.pose_cl = nn.Sequential(
                    nn.Linear(cfg.MEDPOSE.MODEL_DIM * (cfg.MEDPOSE.ROI_MAP_DIM ** 2), 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, 2)
                )
        # heatmap producer network (not linear so possibly more accurate? --> need to test hypothesis)
        self.heatmap_bilinear_interpolation = mynn.BilinearInterpolation2d(cfg.MEDPOSE.NUM_KEYPOINTS, cfg.MEDPOSE.NUM_KEYPOINTS, cfg.MEDPOSE.HEATMAP_UPSCALE)
        # regression network (linear but not quite accurate) 
        # self.pose_regress = nn.Sequential(
        #             nn.Linear(cfg.MEDPOSE.MODEL_DIM, 64),
        #             nn.ReLU(),
        #             nn.Dropout(0.1),
        #             nn.Linear(64, cfg.MEDPOSE.NUM_KEYPOINTS * 2)
        #         )

        self.apply(self._init_weights)
        self._fix_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.KRCNN.CONV_INIT == 'GaussianFill':
                init.normal(m.weight, std=0.01)
            elif cfg.KRCNN.CONV_INIT == 'MSRAFill':
                init.kaiming_normal(m.weight)
            else:
                ValueError('Unexpected cfg.KRCNN.CONV_INIT: {}'.format(cfg.KRCNN.CONV_INIT))
            init.constant(m.bias, 0)

    def _fix_weights(self):
        if cfg.KRCNN.CONV_INIT == 'GaussianFill':
            init.normal(self.conv_fcn_kp.weight, std=0.001)
        elif cfg.KRCNN.CONV_INIT == 'MSRAFill':
            init.kaiming_normal(self.conv_fcn_kp.weight)
        else:
            raise ValueError(cfg.KRCNN.CONV_INIT)
        init.constant(self.conv_fcn_kp.bias, 0)
    
    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        for i in range(cfg.KRCNN.NUM_STACKED_CONVS):
            detectron_weight_mapping['conv_fcn_fm.%d.weight' % (2*i)] = 'conv_fcn_fm%d_w' % (i+1)
            detectron_weight_mapping['conv_fcn_fm.%d.bias' % (2*i)] = 'conv_fcn_fm%d_b' % (i+1)
            detectron_weight_mapping['conv_fcn_rf.%d.weight' % (2*i)] = 'conv_fcn_rf%d_w' % (i+1)
            detectron_weight_mapping['conv_fcn_rf.%d.bias' % (2*i)] = 'conv_fcn_rf%d_b' % (i+1)

        blob_name = 'kps_score_lowres'
        detectron_weight_mapping.update({
            'upsample.upconv.weight': None,  # 0: don't load from or save to checkpoint
            'upsample.upconv.bias': None
        })
        
        detectron_weight_mapping.update({
            'conv_fcn_kp.weight': blob_name + '_w',
            'conv_fcn_kp.bias': blob_name + '_b'
        })
        return detectron_weight_mapping, orphan_in_detectron
    
    def _reformat_region_feats(self, region_features, rois, batch_size, device_id):
        # reformat region to (batch_size x max_regions x ...), with padding, from (total_regions x ...)
        max_regions = 300
        #max_regions = max([(rois[:,0] == idx).nonzero().shape[0] for idx in range(batch_size)])
        default_shape = list(region_features.shape[1:])
        reformatted_region_feats = Variable(torch.zeros(batch_size, max_regions, *default_shape)).cuda(device_id)
        # for each batch, fill zero tensor with correct region data
        batch_dict = {}
        for batch_idx in range(batch_size):
            indices = np.where(rois[:,0] == batch_idx)[0].tolist()
            batch_dict[batch_idx] = len(indices)
            if len(indices) != 0:
                curr_region_feat = region_features[indices]
                num_regions = curr_region_feat.shape[0]
                reformatted_region_feats[batch_idx, :num_regions] = curr_region_feat
        return reformatted_region_feats, batch_dict
    
    def _reconstruct_outputs(self, out, batch_dict):
        new_out = []
        for batch_idx in batch_dict:
            if batch_dict[batch_idx] > 0:
                new_out.append(out[batch_idx, :batch_dict[batch_idx]])
        return torch.cat(new_out, dim=0)
   
    def forward(self, x, rpn_ret, pose_detections=[], initial_frame=True):
        # currently one scale feature map (largest one)
        feature_map = x[-1]
        feature_map = self.conv_fcn_fm(feature_map)
        device_id = feature_map.get_device()
        # roi_idx = len(x) - 1 + 2
        blob_roi_str = 'keypoint_rois'
        # blob_roi_str = 'keypoint_rois_fpn{}'.format(roi_idx)
        region_features = self.roi_xform(
            x, rpn_ret,
            blob_rois=blob_roi_str,
            method=cfg.KRCNN.ROI_XFORM_METHOD,
            resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        # testing stuff
        region_features = self.conv_fcn_rf(region_features)
        residual_connection = self.conv_fcn_kp(region_features) # used for later learning signal reinforcement (can share conv weights for region feature feature enlargement)
        
        # real stuff
        formatted_region_features, batch_dict = self._reformat_region_feats(region_features, rpn_ret[blob_roi_str], feature_map.shape[0], device_id)
        #formatted_region_features = formatted_region_features.view(*formatted_region_features.shape[:-2], -1).permute(0, 2, 1, 3)
        #formatted_region_features = self.conv_fcn_rf(formatted_region_features)
        #formatted_region_features = formatted_region_features.view(*formatted_region_features.shape[:-1], cfg.KRCNN.ROI_XFORM_RESOLUTION, cfg.KRCNN.ROI_XFORM_RESOLUTION).permute(0, 2, 1, 3, 4)
        #print("inputs into encoder stack", feature_map.unsqueeze(dim=1).shape, formatted_region_features.shape)
        enc_out = self.encoder(feature_map.unsqueeze(dim=1), formatted_region_features.contiguous(), initial_frame, True)
        #print("output of encoder stack", enc_out.shape)
        enc_out = self._reconstruct_outputs(enc_out, batch_dict)
        enc_out = enc_out.view(enc_out.shape[0], cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.ROI_XFORM_RESOLUTION, cfg.KRCNN.ROI_XFORM_RESOLUTION)
        # residual connection to enforce learning (viewing into heatmap doesn't take into account spatial information --> still have to test this theory with ablation)
        heat_out = enc_out + residual_connection
        #heat_out = enc_out
        # classifier (whether they contribute to human poses or not) -> not currently used
        # cl_in = region_features.view(region_features.shape[0], -1)
        # cl_out = self.pose_cl(cl_in)
        # new strategy: produce heatmaps
        heat_out = self.heatmap_bilinear_interpolation(heat_out)
        # old strategy: regress keypoints
        #reg_out = self.pose_regress(enc_out)
        #reg_out = reg_out.view(reg_out.shape[0], cfg.MEDPOSE.NUM_KEYPOINTS, 2).permute(0,2,1)
        #return reg_out, cl_out
        return heat_out, None

    ################ THIS OUTPUTS DATA SIMILAR TO WHAT WE SAW IN FIRST IMPL #################
    # def forward(self, feature_map, region_features, rois, pose_detections=[], initial_frame=True):
    #     device_id = feature_map.get_device()
    #     region_features = self._reformat_region_feats(region_features, rois, feature_map.shape[0], device_id)
    #     enc_out = self.encoder(feature_map.unsqueeze(dim=1), region_features, initial_frame, True)
    #     cl_in = region_features.view(region_features.shape[0], region_features.shape[1], -1)
    #     cl_out = self.pose_cl(cl_in)
    #     reg_out = self.pose_regress(enc_out)
    #     return reg_out, cl_out

def keypoint_losses(kps_pred, keypoint_locations_int32, keypoint_weights, keypoint_loss_normalizer=None):
    #kps_targets = []
    #kps_preds = []
    #kps_pred = kps_pred.view(kps_pred.shape[0], cfg.MEDPOSE.NUM_KEYPOINTS, 2).permute(0,2,1)
    #potentially use heatmaps of keypoints too for alternate loss
    #visibilities = keypoint_locations_vec[:, 2]
    #pred_heatmaps = keypoints_to_pred_heatmap(kps_pred, kps_rois, visibilities)
    # new
    device_id = kps_pred.get_device()
    kps_target = Variable(torch.from_numpy(
                keypoint_locations_int32.astype('int64').squeeze())).cuda(device_id)
    keypoint_weights = Variable(torch.from_numpy(keypoint_weights.squeeze())).cuda(device_id)
    losses = F.cross_entropy(kps_pred.view(-1, cfg.KRCNN.HEATMAP_SIZE**2), kps_target, reduce=False)
    loss = cfg.KRCNN.LOSS_WEIGHT * torch.sum(losses * keypoint_weights) / torch.sum(keypoint_weights)
    # old 
    # device_id = kps_pred.get_device()
    # for batch_idx in range(keypoint_locations_vec.shape[0]):
    #     vis_ind = np.where(keypoint_locations_vec[batch_idx, 2] > 0)[0].tolist()
    #     mask = torch.zeros(cfg.MEDPOSE.NUM_KEYPOINTS)
    #     mask[vis_ind] = 1
    #     kps_targets.append(torch.from_numpy(keypoint_locations_vec[batch_idx, :2]).float() * mask)
    #     kps_preds.append(kps_pred[batch_idx] * Variable(mask).cuda(device_id))
    # losses = []
    # for idx in range(len(kps_targets)):
    #     losses.append(F.mse_loss(kps_pred[idx], Variable(kps_targets[idx]).cuda(device_id)))
    # losses = torch.stack(losses)
    # loss = cfg.KRCNN.LOSS_WEIGHT * torch.sum(losses)

    if not cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        # Discussion: the softmax loss above will average the loss by the sum of
        # keypoint_weights, i.e. the total number of visible keypoints. Since
        # the number of visible keypoints can vary significantly between
        # minibatches, this has the effect of up-weighting the importance of
        # minibatches with few visible keypoints. (Imagine the extreme case of
        # only one visible keypoint versus N: in the case of N, each one
        # contributes 1/N to the gradient compared to the single keypoint
        # determining the gradient direction). Instead, we can normalize the
        # loss by the total number of keypoints, if it were the case that all
        # keypoints were visible in a full minibatch. (Returning to the example,
        # this means that the one visible keypoint contributes as much as each
        # of the N keypoints.)
        loss *= keypoint_loss_normalizer.item() # np.float32 to float
    return loss
