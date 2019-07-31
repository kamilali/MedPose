from components.mp_base import MedPoseBase
from components.mp_encoder import MedPoseEncoder
from components.mp_decoder import MedPoseDecoder
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt

class MedPose(nn.Module):

    def __init__(self, num_keypoints=17, num_rpn_props=300):
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
        self.encoder = MedPoseEncoder(num_enc_layers=2)
        '''
        initialize the MedPose decoder architecture with
        default parameters (decoder attends to local structures
        from previous pose estimations and uses encoder outputs
        as queries for subsequent pose detections)
        '''
        self.decoder = MedPoseDecoder(num_dec_layers=2)
    
    def forward(self, x, pose_detections=[]):
        '''
        set to eval mode since the base cnn model takes different
        inputs based on the mode and we are not fine tuning
        '''
        self.base.eval()
        '''
        extract feature maps and region proposals using base
        model as well as region features using extracted feature
        maps and region proposals for current frame
        '''
        with torch.no_grad():
            feature_maps, region_features = self.base.extract_base_features(x)
        feature_maps = feature_maps[0]
        # visualize a single channel feature map
        #plt.imshow(feature_maps[0].detach().cpu().squeeze(dim=0).numpy()[1,:,:])
        #plt.show()
        # visualize a single channel region feature map
        #plt.imshow(region_features[0].detach().cpu().squeeze(dim=0).numpy()[0,:,:])
        #plt.show()
        
        cf_region_features = region_features[-self.num_rpn_props:]

        enc_out = self.encoder(feature_maps.unsqueeze(dim=0), cf_region_features.unsqueeze(dim=0))

        print("ENCODER OUTPUT: ", enc_out.shape)
        
        if len(pose_detections) == 0:
            curr_pose_estimation, curr_pose_classes = self.decoder(enc_out)
        else:
            curr_pose_estimation, curr_pose_classes = self.decoder(enc_out, 
                    torch.stack(pose_detections, dim=1))
        print("DECODER OUTPUTS: ", curr_pose_estimation.shape, curr_pose_classes.shape)

        return curr_pose_estimation
