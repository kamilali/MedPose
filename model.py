from components.mp_base import MedPoseBase
from components.mp_encoder import MedPoseEncoder
from components.mp_decoder import MedPoseDecoder
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt

class MedPose(nn.Module):

    def __init__(self, device, gpus, num_keypoints=17, num_rpn_props=300, window_size=5):
        super(MedPose, self).__init__()
        '''
        initialize the MedPose base architecture to extract
        feature maps and region proposals
        '''
        self.base = MedPoseBase(num_keypoints=num_keypoints, num_rpn_props=num_rpn_props).to(device)
        self.num_rpn_props = num_rpn_props
        '''
        initialize the MedPose encoder architecture with
        default parameters (encoder attends to local structures
        using feature maps bounded by region proposals as
        queries)
        '''
        self.encoder = nn.DataParallel(MedPoseEncoder(num_enc_layers=4, lrnn_window_size=window_size), device_ids=gpus[1:]).to(gpus[1])
        self.window_size = window_size
        '''
        initialize the MedPose decoder architecture with
        default parameters (decoder attends to local structures
        from previous pose estimations and uses encoder outputs
        as queries for subsequent pose detections)
        '''
        self.decoder = nn.DataParallel(MedPoseDecoder(num_dec_layers=4, lrnn_window_size=window_size), device_ids=gpus[1:]).to(gpus[1])
    
    def forward(self, x, pose_detections=[], initial_frame=True):
        '''
        set to eval mode since the base cnn model takes different
        inputs based on the mode and we are not fine tuning
        '''
        self.base.eval()
        '''
        iterate through frames and process batch of frames in parallel
        '''
        frame_batches = list(map(list, zip(*x)))
        x = []
        pose_detections = []
        pose_classifications = []
        initial_frame = True

        for frame_batch in frame_batches:
            x.append(frame_batch)
            x = x[-self.window_size:]
            base_in = list(map(list, zip(*x)))
            '''
            extract feature maps and region proposals using base
            model as well as region features using extracted feature
            maps and region proposals for current frame
            '''
            with torch.no_grad():
                feature_maps, cf_region_features = self.base.extract_base_features(base_in)

            enc_out = self.encoder(feature_maps, cf_region_features, initial_frame)
            
            del feature_maps
            del cf_region_features
            torch.cuda.empty_cache()
            
            if len(pose_detections) == 0:
                curr_pose_estimation, curr_pose_classes = self.decoder(enc_out, None, initial_frame)
            else:
                curr_pose_estimation, curr_pose_classes = self.decoder(enc_out, 
                        torch.stack(pose_detections, dim=1), initial_frame)
            
            pose_detections.append(curr_pose_estimation)
            pose_detections = pose_detections[-self.window_size:]
            pose_classifications.append(curr_pose_classes)
            pose_classifications = pose_classifications[-self.window_size:]
            initial_frame = False

            del enc_out
            del curr_pose_estimation
            del curr_pose_classes
            torch.cuda.empty_cache()

        return pose_detections, pose_classifications
