import torch.nn as nn
import torch
from .mp_layers import MedPoseAttention, MedPoseConvLSTM

class MedPoseEncoder(nn.Module):

    def __init__(self, num_enc_layers=3, num_att_heads=4, num_lrnn_layers=3, 
            model_dim=256, lrnn_hidden_dim=256, ff_hidden_dim=1024, 
            roi_map_dim=7, lrnn_window_size=3):
        super(MedPoseEncoder, self).__init__()
        '''
        store number of encoder layers and a dictionary containing
        history of outputs per encoder layer for recurrence
        Also create module list for all sub-layers to keep track of 
        them for each encoder layer
        '''
        self.num_enc_layers = num_enc_layers
        self.hist = {}

        self.local_rnns = nn.ModuleList()
        self.lrnn_layer_norms = nn.ModuleList()
        self.lrnn_window_size = lrnn_window_size
        self.hidden_rnn_hist = {}

        self.atts = nn.ModuleList()
        self.att_layer_norms = nn.ModuleList()

        self.ffs = nn.ModuleList()
        self.ff_layer_norms = nn.ModuleList()

        for enc_layer in range(self.num_enc_layers):
            '''
            initially no history exists for encoder layers
            '''
            self.hist[enc_layer]= []
            '''
            LocalRNN for capturing local structures to attend to
            However, since we are dealing with images, we modify the
            R-Transformer slightly by using a ConvLSTM instead of a 
            LSTM as the LocalRNN module
            '''
            local_rnn = MedPoseConvLSTM(
                    num_layers=num_lrnn_layers, 
                    input_size=model_dim,
                    model_size=model_dim,
                    hidden_size=lrnn_hidden_dim,
                    batch_first=True,
                    conv2d_req=(enc_layer == 0),
                    conv1d_req=(enc_layer != 0))
            self.local_rnns.append(local_rnn)

            lrnn_layer_norm = nn.LayerNorm(lrnn_hidden_dim, eps=1e-05, elementwise_affine=True)
            self.lrnn_layer_norms.append(lrnn_layer_norm)
            '''
            since attention mechanism attends to all local contexts
            for a particular frame, we store the outputs of all local
            rnns for a given video
            '''
            self.hidden_rnn_hist[enc_layer] = []
            '''
            Attention mechanism using region features to obtain queries 
            and outputs of LocalRNNs to obtain keys and values
            '''
            att = MedPoseAttention(
                    query_dim=((roi_map_dim ** 2) * model_dim),
                    key_dim=lrnn_hidden_dim, 
                    value_dim=lrnn_hidden_dim, 
                    model_dim=model_dim,
                    num_att_heads=num_att_heads)
            self.atts.append(att)

            att_layer_norm = nn.LayerNorm(model_dim, eps=1e-05, elementwise_affine=True)
            self.att_layer_norms.append(att_layer_norm)
            '''
            feed-forward module to map output of attention layer to final
            encoder output (the above sub-layers can be stacked through
            multiple encoder layers)
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
    
    def forward(self, feature_maps, cf_region_features, initial_frame=True):
        '''
        check if initial frame of video and clear histories if it is
        '''
        if initial_frame:
            for enc_layer in range(self.num_enc_layers):
                self.hist[enc_layer] = []
                self.hidden_rnn_hist[enc_layer] = []
        '''
        apply LocalRNN to small local window to capture local
        structures and only keep the last hidden state representation
        '''
        seq_len = feature_maps.shape[1]

        if seq_len > self.lrnn_window_size:
            feature_maps = feature_maps[:, -self.lrnn_window_size:]
        
        enc_in = feature_maps
        '''
        stack encoders based on number of encoder layers specified
        '''
        for enc_layer in range(self.num_enc_layers):
            '''
            if processing feature maps, use ConvLSTM with 2d convolutions
            otherwise use ConvLSTM with 1d convolutions and stack sequences
            based on stored hist for current layer
            '''
            if enc_layer == 0:
                (context, _), residual_connection = self.local_rnns[enc_layer](enc_in)
            else:
                enc_in = torch.stack(self.hist[enc_layer], dim=1)
                enc_in = enc_in.permute(0, 1, 3, 2)
                (context, _), residual_connection = self.local_rnns[enc_layer](enc_in)
            '''
            layer normalization + residual connection (comes from output
            of conv layers prior to forward pass through lstm)
            '''
            context = self.lrnn_layer_norms[enc_layer](context + residual_connection)
            '''
            use the last hidden layer as the hidden representation
            '''
            context = context[:, context.shape[1] - 1]
            context = context.unsqueeze(dim=1)
            '''
            add hidden rnn output to history for attending over time
            '''
            self.hidden_rnn_hist[enc_layer].append(context)
            all_context = torch.cat(self.hidden_rnn_hist[enc_layer], dim=1)
            '''
            attend to local structures using region features as
            queries and outputs of LocalRNNs as context
            '''
            x = self.atts[enc_layer](cf_region_features, all_context)
            '''
            layer normalization + residual connection
            '''
            x = self.att_layer_norms[enc_layer](x + context)
            '''
            transform features non-linearly through feed forward
            module
            '''
            enc_out = self.ffs[enc_layer](x)
            '''
            layer normalization + residual connection
            '''
            enc_out = self.ff_layer_norms[enc_layer](enc_out + x)
            '''
            store current output in encoder history for next layer and, 
            if the next layer history is full, shift history to
            make room for new output (current output is input for
            next layer)
            '''
            if enc_layer < (self.num_enc_layers - 1):
                if len(self.hist[enc_layer + 1]) == self.lrnn_window_size:
                    self.hist[enc_layer + 1] = self.hist[enc_layer + 1][1:]
                self.hist[enc_layer + 1].append(enc_out)

        return enc_out

