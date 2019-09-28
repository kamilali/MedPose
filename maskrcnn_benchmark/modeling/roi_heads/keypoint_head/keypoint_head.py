import torch
import torch.nn as nn

from .roi_keypoint_feature_extractors import make_roi_keypoint_feature_extractor
from .roi_keypoint_feature_enlarger import make_roi_keypoint_feature_enlarger
from .roi_keypoint_predictors import make_roi_keypoint_predictor
from .inference import make_roi_keypoint_post_processor
from .loss import make_roi_keypoint_loss_evaluator

from .components.mp_encoder import MedPoseEncoder
from .components.mp_decoder import MedPoseDecoder
from .components.mp_layers import MedPoseHistory

from maskrcnn_benchmark import layers

from .utils import cat

class ROIKeypointHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIKeypointHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_keypoint_feature_extractor(cfg, in_channels)
        if cfg.MODEL.MEDPOSE_ON:
            # used to enlarge feature map feature size
            self.feature_map_enlarger = make_roi_keypoint_feature_enlarger(cfg, in_channels)
            # to build residual connection for heatmap production
            self.conv_fcn1 = layers.Conv2d(cfg.MODEL.MEDPOSE.MODEL_DIM, cfg.MODEL.MEDPOSE.NUM_KEYPOINTS, 1, stride=1, padding=0)
            nn.init.kaiming_normal_(self.conv_fcn1.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(self.conv_fcn1.bias, 0)
            # encoder architecture
            encoder_history = [MedPoseHistory() for i in range(cfg.NUM_GPUS)]
            self.encoder = MedPoseEncoder(
                num_enc_layers=cfg.MODEL.MEDPOSE.STACK_LAYERS,
                enc_history=encoder_history,
                model_dim=cfg.MODEL.MEDPOSE.MODEL_DIM,
                lrnn_hidden_dim=cfg.MODEL.MEDPOSE.MODEL_DIM,
                lrnn_window_size=cfg.MODEL.MEDPOSE.WINDOW_SIZE,
                roi_map_dim=cfg.MODEL.MEDPOSE.ROI_MAP_DIM,
                gpus=range(cfg.NUM_GPUS),
                lrnn_batch_norm=False)
            self.lrnn_window_size = cfg.MODEL.MEDPOSE.WINDOW_SIZE
            # heatmap builder (can replace with just the interpolation)
            self.heatmap_builder = make_roi_keypoint_predictor(
                cfg, cfg.MODEL.MEDPOSE.NUM_KEYPOINTS, cfg.MODEL.MEDPOSE.HEATMAP_UPSCALE)
        else:
            self.predictor = make_roi_keypoint_predictor(
                cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_keypoint_post_processor(cfg)
        self.loss_evaluator = make_roi_keypoint_loss_evaluator(cfg)
    
    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois
    
    def _reformat_region_feats(self, region_features, rois, vid_shape): # reformat region to (batch_size x max_regions x ...), with padding, from (total_regions x ...) to process in batch max_regions = 300
        batch_size = vid_shape[0]
        seq_len = vid_shape[1]
        max_regions = 300
        default_shape = list(region_features.shape[1:])
        reformatted_region_feats = torch.zeros((batch_size, max_regions, *default_shape)).to(region_features.get_device())
        # for each batch, fill zero tensor with correct region data
        restore_dict = {}
        for batch_idx in range(batch_size):
            indices = (rois[:,0] == batch_idx).nonzero().squeeze(dim=1)
            restore_dict[batch_idx] = indices.shape[0]
            if indices.shape[0] != 0:
                curr_region_feat = region_features[indices]
                num_regions = curr_region_feat.shape[0]
                reformatted_region_feats[batch_idx, :num_regions] = curr_region_feat
        return reformatted_region_feats, restore_dict
    
    def _reconstruct_outputs(self, out, restore_dict):
        new_out = []
        for batch_idx in restore_dict:
            if restore_dict[batch_idx] > 0:
                new_out.append(out[batch_idx, :restore_dict[batch_idx]])
        # can't think of a better way to handle this yet
        if len(new_out) == 0:
            print("encountered reconstruction with no regions... treating manually")
            return out[0,:1]
        return torch.cat(new_out, dim=0)

    def forward(self, features, proposals, targets=None, vid_shape=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        if self.cfg.MODEL.MEDPOSE_ON:
            preds = []
            loss_proposals = proposals
            ablation = False # if we are ablating, this flag is set to true
            if ablation:
                # not using encoder output, simply passing the residual connection through the heatmap builder
                x = self.feature_extractor(features, proposals)
                residual_connection = self.conv_fcn1(x)
                kp_logits = self.heatmap_builder(residual_connection)
            else:
                proposals = [proposals[(i * vid_shape[1]):(i * vid_shape[1] + vid_shape[1])] for i in range(vid_shape[0])]
                # we iterate by seq x batch_len
                proposals = list(zip(*proposals))
                transformed_features = [feature.view(*vid_shape, feature.shape[1], feature.shape[2], feature.shape[3]) for feature in features]
                vid_feat_seq = []
                region_feats_vec = []
                for curr_frame_idx in range(vid_shape[1]):
                    # get current frame proposals across batch of videos
                    proposal = proposals[curr_frame_idx]
                    # get current frame feature maps across batch of videos and per FPN level
                    cf_feature_maps = [transformed_feature[:,curr_frame_idx] for transformed_feature in transformed_features]
                    # extract region features for encoder
                    cf_x = self.feature_extractor(cf_feature_maps, proposal)
                    region_feats_vec.append(cf_x)
                    # residual connection to enforce learning of spatial heatmap from output of encoder
                    residual_connection = self.conv_fcn1(cf_x)
                    # reformat region features to encoder input format (will be reformatted agian later)
                    rois = self.convert_to_roi_format(proposal)
                    region_features, restore_dict = self._reformat_region_feats(cf_x, rois, vid_shape)
                    # get fine-grain feature map for encoder attention (currently operates on single feature level)
                    # enlarge feature map to produce heatmap later
                    # TODO: Assess FPN level attentions for regions across levels of FPN
                    cf_feature_map = cf_feature_maps[0]
                    cf_feature_map = self.feature_map_enlarger(cf_feature_map)
                    # get feature map from video feature map sequence (we only care about the local window)
                    vid_feat_seq.append(cf_feature_map)
                    vid_feat_seq = vid_feat_seq[-self.cfg.MODEL.MEDPOSE.WINDOW_SIZE:]
                    feature_map = torch.stack(vid_feat_seq, dim=1)
                    # get output of encoder (heatmap producing features)
                    enc_out = self.encoder(feature_map, region_features, (curr_frame_idx == 0), True)
                    enc_out = self._reconstruct_outputs(enc_out, restore_dict)
                    enc_out = enc_out.view(enc_out.shape[0], self.cfg.MODEL.MEDPOSE.NUM_KEYPOINTS, self.cfg.MODEL.MEDPOSE.ROI_MAP_DIM, self.cfg.MODEL.MEDPOSE.ROI_MAP_DIM)
                    # (viewing into heatmap doesn't take into account spatial information --> still have to test this theory with ablation) 
                    kp_logits = self.heatmap_builder(enc_out + residual_connection)
                    preds.append(kp_logits)
                    ## no ConvTranspose2d...just the interpolation
                    # kp_logits = layers.interpolate(
                    #     enc_out, scale_factor=self.up_scale, mode="bilinear", align_corners=False
                    # )
                kp_logits = torch.cat(preds, dim=0)
                x = torch.cat(region_feats_vec, dim=0)
                proposals = loss_proposals
        else:
            x = self.feature_extractor(features, proposals)
            kp_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(kp_logits, proposals)
            return x, result, {}
        
        print("GOT HERE...")
        return x, proposals, dict(loss_kp=loss_kp)


def build_roi_keypoint_head(cfg, in_channels):
    return ROIKeypointHead(cfg, in_channels)
