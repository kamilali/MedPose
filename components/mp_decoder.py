import torch.nn as nn
import torch
from .mp_layers import MedPoseAttention, MedPoseConvLSTM

class MedPoseDecoder(nn.Module):

    def __init__(self, num_dec_layers=3, num_att_heads=2, num_lrnn_layers=3,
            model_dim=256, lrnn_hidden_dim=256, ff_hidden_dim=1024,
            roi_map_dim=7, lrnn_window_size=3, num_keypoints=17):
        super(MedPoseDecoder, self).__init__()
        '''
        store number of decoder layers and a dictionary containing
        history of outputs per encoder layer for recurrence
        Also create module list for all sub-layers to keep track of 
        them for each decoder layer
        '''
        self.num_dec_layers = num_dec_layers
        self.hist = {}

        self.local_rnns = nn.ModuleList()
        self.lrnn_layer_norms = nn.ModuleList()
        self.lrnn_window_size = lrnn_window_size

        self.self_atts = nn.ModuleList()
        self.self_att_layer_norms = nn.ModuleList()

        self.enc_dec_atts = nn.ModuleList()
        self.enc_dec_att_layer_norms = nn.ModuleList()

        self.ffs = nn.ModuleList()
        self.ff_layer_norms = nn.ModuleList()

        for dec_layer in range(self.num_dec_layers):
            self.hist[dec_layer]= []
            '''
            LocalRNN for capturing local structures to attend to
            However, since we are dealing with images, we modify the
            R-Transformer slightly by using a ConvLSTM instead of a 
            LSTM as the LocalRNN module
            '''
            local_rnn = MedPoseConvLSTM(
                    num_layers=num_lrnn_layers, 
                    input_size=(num_keypoints * 2),
                    model_size=model_dim,
                    hidden_size=lrnn_hidden_dim,
                    batch_first=True,
                    conv2d_req=False,
                    conv1d_req=True,
                    decoder_first=(dec_layer == 0))
            self.local_rnns.append(local_rnn)

            lrnn_layer_norm = nn.LayerNorm(lrnn_hidden_dim, eps=1e-05, elementwise_affine=True)
            self.lrnn_layer_norms.append(lrnn_layer_norm)
            '''
            Attention mechanism (self-attention) on outputs of
            LocalRNN to find local structures to attend to from
            pose detections
            '''
            self_att = MedPoseAttention(
                    query_dim=model_dim,
                    key_dim=lrnn_hidden_dim, 
                    value_dim=lrnn_hidden_dim, 
                    model_dim=model_dim,
                    num_att_heads=num_att_heads)
            self.self_atts.append(self_att)

            self_att_layer_norm = nn.LayerNorm(model_dim, eps=1e-05, elementwise_affine=True)
            self.self_att_layer_norms.append(self_att_layer_norm)
            '''
            Attention mechanism using output of encoder as context
            and using output of prior attention layer as queries
            '''
            enc_dec_att = MedPoseAttention(
                    query_dim=model_dim,
                    key_dim=lrnn_hidden_dim, 
                    value_dim=lrnn_hidden_dim, 
                    model_dim=model_dim,
                    num_att_heads=num_att_heads)
            self.enc_dec_atts.append(enc_dec_att)

            enc_dec_att_layer_norm = nn.LayerNorm(model_dim, eps=1e-05, elementwise_affine=True)
            self.enc_dec_att_layer_norms.append(enc_dec_att_layer_norm)
            '''
            feed-forward module to map output of attention layer to final
            decoder output (the above sub-layers can be stacked through
            multiple decoder layers)
            '''
            ff = nn.Sequential(
                        nn.Linear(model_dim, ff_hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(ff_hidden_dim, model_dim)
                    )
            self.ffs.append(ff)

            ff_layer_norm = nn.LayerNorm(model_dim, eps=1e-05, elementwise_affine=True)
            self.ff_layer_norms.append(ff_layer_norm)
        '''
        fully connected networks for classification (pose detectable
        or not) and regression (regressing from final decoder output
        to joint coordinates per region)
        '''
        self.pose_cl = nn.Sequential(
                    nn.Linear(model_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, 1)
                )
        self.pose_regress = nn.Sequential(
                    nn.Linear(model_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, num_keypoints * 2)
                )
    
    def forward(self, enc_out, poses=None):
        '''
        stack decoders based on number of decoder layers specified
        '''
        for dec_layer in range(self.num_dec_layers):
            if poses is None and dec_layer == 0:
                '''
                use encoder as query and context for first pose detection
                (skip the first two layers of the decoder since those
                require prior pose estimations)
                '''
                eda_out = self.enc_dec_atts[dec_layer](enc_out, enc_out)
                '''
                layer normalization +  residual connection
                '''
                eda_out = self.enc_dec_att_layer_norms[dec_layer](eda_out + enc_out)
                '''
                transform features non-linearly through feed forward
                module
                '''
                dec_out = self.ffs[dec_layer](eda_out)
                '''
                layer normalization + residual connection
                '''
                dec_out = self.ff_layer_norms[dec_layer](dec_out + eda_out)
            else:
                if dec_layer == 0:
                    seq_len = poses.shape[1]
                    if seq_len > self.lrnn_window_size:
                        poses = poses[:, -self.lrnn_window_size:]
                    poses = poses.permute(0, 1, 3, 2)
                    (context, _), residual_connection = self.local_rnns[dec_layer](poses)
                else:
                    dec_in = torch.stack(self.hist[dec_layer], dim=1)
                    dec_in = dec_in.permute(0, 1, 3, 2)
                    (context, _), residual_connection = self.local_rnns[dec_layer](dec_in)
                '''
                layer normalization + residual connection (comes from output
                of conv layers prior to forward pass through lstm)
                '''
                context = self.lrnn_layer_norms[dec_layer](context + residual_connection)
                '''
                use the last hidden layer as the hidden representation
                '''
                context = context[:, context.shape[1] - 1]
                context = context.unsqueeze(dim=1)
                '''
                attend to local structures using hidden representation
                as query and context for self attention
                '''
                sa_out = self.self_atts[dec_layer](context, context)
                '''
                layer normalization + residual connection
                '''
                sa_out = self.self_att_layer_norms[dec_layer](sa_out + context)
                '''
                attend to local structures using output of prior
                attention layer as queries and the corresponding
                encoder output as the context
                '''
                eda_out = self.enc_dec_atts[dec_layer](sa_out, enc_out)
                '''
                layer normalization + residual connection
                '''
                eda_out = self.enc_dec_att_layer_norms[dec_layer](eda_out + enc_out)
                '''
                transform features non-linearly through feed forward
                module
                '''
                dec_out = self.ffs[dec_layer](eda_out)
                '''
                layer normalization + residual connection
                '''
                dec_out = self.ff_layer_norms[dec_layer](dec_out + eda_out)
            '''
            store current output in decoder history for next layer and, 
            if the next layer history is full, shift history to
            make room for new output (current output is input for
            next layer)
            '''
            if dec_layer < (self.num_dec_layers - 1):
                if len(self.hist[dec_layer + 1]) == self.lrnn_window_size:
                    self.hist[dec_layer + 1] = self.hist[dec_layer + 1][1:]
                self.hist[dec_layer + 1].append(dec_out)
        '''
        pass output of last decoder layer to fully connected network for
        pose estimation
        '''
        curr_poses_classes = self.pose_cl(dec_out)
        curr_poses = self.pose_regress(dec_out)

        return curr_poses, curr_poses_classes

