import torch.nn as nn
import torch
from math import sqrt

class MedPoseAttention(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, model_dim, num_att_heads):
        super(MedPoseAttention, self).__init__()
        '''
        linear transformations to map inputs into query and context
        vectors for attention computation. Linear projections per head
        are stacked into a ModuleList
        '''
        self.model_dim = model_dim // num_att_heads
        self.query_mappers = nn.ModuleList(
                    [nn.Linear(query_dim, self.model_dim) for _ in range(num_att_heads)]
                )
        self.key_mappers = nn.ModuleList(
                    [nn.Linear(key_dim, self.model_dim) for _ in range(num_att_heads)]
                )
        self.value_mappers = nn.ModuleList(
                    [nn.Linear(value_dim, self.model_dim) for _ in range(num_att_heads)]
                )
        self.num_att_heads = num_att_heads
        '''
        project concatenated head attention outputs
        '''
        self.multi_head_att_mapper = nn.Linear(self.num_att_heads * self.model_dim, model_dim)
        '''
        softmax layer for attention computation
        '''
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries, context):
        '''
        map inputs to query, key, and value vectors
        '''
        queries = queries.view(queries.shape[0], queries.shape[1], -1)
        q = self.query_mappers[0](queries)
        k = self.key_mappers[0](context)
        v = self.value_mappers[0](context)
        '''
        attention computation based on Q, K, V - linear projections 
        of inputs
        '''
        att_weights = self.softmax(torch.div(torch.matmul(q, torch.transpose(k, 1, 2)), sqrt(self.model_dim)))
        multi_head_att_out = torch.matmul(att_weights, v)
        residual_connection = q

        for idx in range(1, self.num_att_heads):
            q = self.query_mappers[idx](queries)
            k = self.key_mappers[idx](context)
            v = self.value_mappers[idx](context)

            att_weights = self.softmax(
                    torch.div(torch.matmul(q, torch.transpose(k, 1, 2)), sqrt(self.model_dim)))
            multi_head_att_out = torch.cat([multi_head_att_out, torch.matmul(att_weights, v)], dim=2)
            residual_connection = torch.cat([residual_connection, q], dim=2)
        
        return self.multi_head_att_mapper(multi_head_att_out), residual_connection

class MedPoseConvLSTM(nn.Module):
    
    def __init__(self, num_layers, input_size, model_size, hidden_size, batch_first=False, conv2d_req=True, conv1d_req=False, decoder_first=False):
        super(MedPoseConvLSTM, self).__init__()
        if conv2d_req:
            '''
            process image inputs with convolutions so that they can be 
            passed into an LSTM to capture dependencies over frame
            sequences
            '''
            self.conv2d = nn.Sequential(
                        nn.Conv2d(input_size, input_size // 2, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm2d(
                            input_size // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(input_size // 2, input_size // 4, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm2d(
                            input_size // 4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(input_size // 4, input_size // 2, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm2d(
                            input_size // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(input_size // 2, input_size, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm2d(
                            input_size, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2)
                    )

        if conv1d_req:
            '''
            in processing outputs of encoder layers instead of feature
            map inputs, 2d convolutions are substituted with 1d convolutions
            '''
            if decoder_first:
                self.conv1d = nn.Sequential(
                            nn.Conv1d(input_size, input_size // 2, kernel_size=5, padding=1, stride=2),
                            nn.BatchNorm1d(
                                input_size // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size=2),
                            nn.Conv1d(input_size // 2, input_size // 4, kernel_size=5, padding=1, stride=1),
                            nn.BatchNorm1d(
                                input_size // 4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size=2),
                            nn.Conv1d(input_size // 4, input_size // 4, kernel_size=5, padding=0, stride=1),
                            nn.BatchNorm1d(
                                input_size // 4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU()
                        )
            else:

                self.conv1d = nn.Sequential(
                            nn.Conv1d(model_size, model_size // 2, kernel_size=5, padding=1, stride=2),
                            nn.BatchNorm1d(
                                model_size // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size=2),
                            nn.Conv1d(model_size // 2, model_size // 4, kernel_size=5, padding=1, stride=2),
                            nn.BatchNorm1d(
                                model_size // 4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size=2),
                            nn.Conv1d(model_size // 4, model_size // 8, kernel_size=5, padding=1, stride=2),
                            nn.BatchNorm1d(
                                model_size // 8, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU()
                        )
        self.conv2d_req = conv2d_req
        self.conv1d_req = conv1d_req
        '''
        LSTM layer to process output of conv layers (this is the 
        recurrent part of the ConvLSTM architecture to model image 
        sequences)
        '''
        self.lstm = nn.LSTM(
                num_layers=3, 
                input_size=model_size, hidden_size=hidden_size,
                batch_first=True)
        self.batch_first = batch_first

    def forward(self, x):
        '''
        ensure that batch dimension is first (canonical format
        for consistency)
        '''
        if not self.batch_first:
            if self.conv2d_req and conv2d:
                x = x.permute(1, 0, 2, 3, 4)
            else:
                x = x.permute(1, 0, 2, 3)
        '''
        apply convolution operations before passing
        through lstm to capture dependencies
        '''
        if self.conv2d_req:
            batch_size, num_frames, c, h, w = x.shape
            c_in = x.view(batch_size * num_frames, c, h, w)
            c_out = self.conv2d(c_in)
            r_in = c_out.view(batch_size, num_frames, -1)
        elif self.conv1d_req:
            batch_size, num_frames, c, l = x.shape
            c_in = x.view(batch_size * num_frames, c, l)
            c_in = c_in.to(next(self.conv1d.parameters()).get_device())
            c_out = self.conv1d(c_in)
            r_in = c_out.view(batch_size, num_frames, -1)
        else:
            r_in = x
        self.lstm.flatten_parameters()
        return self.lstm(r_in), r_in

class MedPoseHistory:
    def __init__(self):
        self.history = {}
        self.history_device = {}
        self.lrnn_history = {}
        self.lrnn_history_device = {}
   
    def get_history_size(self, layer):
        if layer not in self.history:
            return 0
        return len(self.history[layer])

    def set_history(self, layer, item):
        self.history[layer] = item

    def append_history(self, layer, item):
        if layer not in self.history:
            self.history[layer] = []
        self.history[layer].append(item)
    
    def set_history_device(self, layer, device):
        self.history_device[layer] = device
    
    def set_lrnn_history(self, layer, item):
        self.lrnn_history[layer] = item
    
    def set_lrnn_history_device(self, layer, device):
        self.lrnn_history_device[layer] = device
    
    def get_history(self, layer):
        if layer not in self.history:
            return None
        return self.history[layer]
    
    def get_history_device(self, layer):
        if layer not in self.history_device:
            return None
        return self.history_device[layer]
    
    def get_lrnn_history(self, layer):
        if layer not in self.lrnn_history:
            return None
        return self.lrnn_history[layer]
    
    def get_lrnn_history_device(self, layer):
        if layer not in self.lrnn_history_device:
            return None
        return self.lrnn_history_device[layer]
    
    def reset(self):
        self.history = {}
        self.history_device = {}
        self.lrnn_history = {}
        self.lrnn_history_device = {}
