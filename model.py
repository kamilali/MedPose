from components.mp_base import MedPoseBase
from components.mp_encoder import MedPoseEncoder
from components.mp_decoder import MedPoseDecoder
from components.mp_layers import MedPoseHistory
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
import time

class MedPose(nn.Module):

    def __init__(self, device, gpus, num_keypoints=17, num_rpn_props=300, window_size=5, stack_layers=4):
        super(MedPose, self).__init__()
        '''
        initialize the MedPose base architecture to extract
        feature maps and region proposals
        '''
        self.base = MedPoseBase(num_keypoints=num_keypoints, num_rpn_props=num_rpn_props)
        self.num_rpn_props = num_rpn_props
        '''
        initialize the MedPose encoder architecture with
        default parameters (encoder attends to local structures
        using feature maps bounded by region proposals as
        queries)
        '''
        encoder_history = [MedPoseHistory() for i in gpus[1:]]
        self.encoder = MedPoseEncoder(num_enc_layers=stack_layers, enc_history=encoder_history, lrnn_window_size=window_size, gpus=gpus[1:], device=device)
        self.window_size = window_size
        '''
        initialize the MedPose decoder architecture with
        default parameters (decoder attends to local structures
        from previous pose estimations and uses encoder outputs
        as queries for subsequent pose detections)
        '''
        decoder_history = [MedPoseHistory() for i in gpus[1:]]
        self.decoder = MedPoseDecoder(num_dec_layers=stack_layers, dec_history=decoder_history, lrnn_window_size=window_size, gpus=gpus[1:], device=device)
    
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
            #print("Processing frame; initial frame status:", initial_frame)
            x.append(frame_batch)
            x = x[-self.window_size:]
            base_in = list(map(list, zip(*x)))
            '''
            extract feature maps and region proposals using base
            model as well as region features using extracted feature
            maps and region proposals for current frame
            '''
            #start_time = time.time() 
            with torch.no_grad():
                feature_maps, cf_region_features = self.base.extract_base_features(base_in)
            #print(time.time() - start_time, "seconds for base")
            #start_time = time.time()
            
            enc_out = self.encoder(feature_maps, cf_region_features, initial_frame)
            #print(time.time() - start_time, "seconds for encoder")
            #torch.cuda.empty_cache()
            
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

            #torch.cuda.empty_cache()

        return pose_detections, pose_classifications
