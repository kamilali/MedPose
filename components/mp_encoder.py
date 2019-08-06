import torch.nn as nn
import torch
from .mp_layers import MedPoseAttention, MedPoseConvLSTM

class MedPoseEncoder(nn.Module):

    def __init__(self, num_enc_layers=3, num_att_heads=4, num_lrnn_layers=3, 
            model_dim=256, lrnn_hidden_dim=256, ff_hidden_dim=1024, 
            roi_map_dim=7, lrnn_window_size=3, gpus=None, device=None):
        super(MedPoseEncoder, self).__init__()
        '''
        store number of encoder layers and a dictionary containing
        history of outputs per encoder layer for recurrence
        Also create module list for all sub-layers to keep track of 
        them for each encoder layer
        '''
        self.num_enc_layers = num_enc_layers
        self.hist = {}
        self.hist_device = {}

        self.local_rnns = nn.ModuleList()
        self.lrnn_layer_norms = nn.ModuleList()
        self.lrnn_window_size = lrnn_window_size
        self.hidden_rnn_hist = {}
        self.hr_hist_device = {}

        self.atts = nn.ModuleList()
        self.att_layer_norms = nn.ModuleList()

        self.ffs = nn.ModuleList()
        self.ff_layer_norms = nn.ModuleList()
        '''
        class variables for parallelism
        '''
        self.gpus = gpus
        self.device = gpus[0]
        '''
        initialize stack layers
        '''
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
            self.hidden_rnn_hist[enc_layer] = None
            '''
            Attention mechanism using region features and current context
            to obtain queries and outputs of LocalRNNs to obtain keys and
            values
            '''
            query_dim = ((roi_map_dim ** 2) * model_dim)
            att = MedPoseAttention(
                    query_dim=query_dim,
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
            self.hist = {}
            self.hist_device = {}
            self.hidden_rnn_hist = {}
            self.hr_hist_device = {}
            for enc_layer in range(self.num_enc_layers):
                self.hist[enc_layer] = []
                self.hidden_rnn_hist[enc_layer] = None
                self.hist_device[enc_layer] = None
                self.hr_hist_device[enc_layer] = None
        '''
        apply LocalRNN to small local window to capture local
        structures and only keep the last hidden state representation
        '''
        enc_in = feature_maps[:, -self.lrnn_window_size:]
        '''
        stack encoders based on number of encoder layers specified
        '''
        for enc_layer in range(self.num_enc_layers):
            #input("entered encoder layer " + str(enc_layer))
            '''
            if processing feature maps, use ConvLSTM with 2d convolutions
            otherwise use ConvLSTM with 1d convolutions and stack sequences
            based on stored hist for current layer
            '''
            if enc_layer != 0:
                enc_in = torch.stack(self.hist[enc_layer], dim=1)
                enc_in = enc_in.permute(0, 1, 3, 2)
            #(context, _), residual_connection = self.local_rnns[enc_layer](enc_in)
            (context, _), residual_connection = data_parallel(self.local_rnns[enc_layer], enc_in, self.gpus, self.device)
            if enc_layer == 0:
                del feature_maps
            else:
                del enc_in
            torch.cuda.empty_cache()
            '''
            layer normalization + residual connection (comes from output
            of conv layers prior to forward pass through lstm)
            '''
            #context = self.lrnn_layer_norms[enc_layer](context + residual_connection)
            context = data_parallel(self.lrnn_layer_norms[enc_layer], context + residual_connection, self.gpus, self.device)
            '''
            use the last hidden layer as the hidden representation
            '''
            context = context[:, context.shape[1] - 1]
            context = context.unsqueeze(dim=1)
            '''
            add hidden rnn output to history for attending over past 
            local structures
            '''
            if self.hr_hist_device[enc_layer] is None:
                self.hr_hist_device[enc_layer] = context.get_device()
                self.hidden_rnn_hist[enc_layer] = context
            else:
                self.hidden_rnn_hist[enc_layer] = torch.cat((self.hidden_rnn_hist[enc_layer], context.to(self.hr_hist_device[enc_layer])), dim=1)
            '''
            formulate query and past_context for attention
            mechanism
            '''
            query = cf_region_features
            context = self.hidden_rnn_hist[enc_layer]
            '''
            attend to local structures using region features as
            queries and outputs of LocalRNNs as context
            '''
            x, residual_connection = self.atts[enc_layer](query, context)
            '''
            layer normalization + residual connection
            '''
            #x = self.att_layer_norms[enc_layer](x + residual_connection)
            x = data_parallel(self.att_layer_norms[enc_layer], x + residual_connection, self.gpus, self.device)
            '''
            transform features non-linearly through feed forward
            module
            '''
            #enc_out = self.ffs[enc_layer](x)
            enc_out = data_parallel(self.ffs[enc_layer], x, self.gpus, self.device)
            '''
            layer normalization + residual connection
            '''
            #enc_out = self.ff_layer_norms[enc_layer](enc_out + x)
            enc_out = data_parallel(self.ff_layer_norms[enc_layer], enc_out + x, self.gpus, self.device)
            '''
            store current output in encoder history for next layer and, 
            if the next layer history is full, shift history to
            make room for new output (current output is input for
            next layer)
            '''
            if enc_layer < (self.num_enc_layers - 1):
                if self.hist_device[enc_layer + 1] is None:
                    self.hist_device[enc_layer + 1] = enc_out.get_device()
                if len(self.hist[enc_layer + 1]) == self.lrnn_window_size:
                    self.hist[enc_layer + 1] = self.hist[enc_layer + 1][1:]
                self.hist[enc_layer + 1].append(enc_out.to(self.hist_device[enc_layer + 1]))

        return enc_out

def data_parallel(module, x, gpus=None, device=None):
    
    if not gpus:
        if type(x) == list:
            return module(*x)
        else:
            return module(x)

    if device is None:
        device = gpus[0]

    replicas = nn.parallel.replicate(module, gpus)
    inputs = nn.parallel.scatter(x, gpus)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, device)

