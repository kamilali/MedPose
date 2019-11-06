import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributed as dist
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
        self.softmax = nn.Softmax(dim=2)
    
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
        #att_weights = self.softmax(torch.div(torch.matmul(q, torch.transpose(k, 1, 2)), sqrt(self.model_dim)))
        residual_connection = q
        att_weights = self.softmax(torch.div(torch.matmul(q, torch.transpose(k, 1, 2)), sqrt(self.model_dim)))
        multi_head_att_out = torch.matmul(att_weights, v)

        for idx in range(1, self.num_att_heads):
            q = self.query_mappers[idx](queries)
            k = self.key_mappers[idx](context)
            v = self.value_mappers[idx](context)

            #torch.div(torch.matmul(q, torch.transpose(k, 1, 2)), sqrt(self.model_dim)))
            residual_connection = torch.cat([residual_connection, q], dim=2)
            att_weights = self.softmax(
                    torch.div(torch.matmul(q, torch.transpose(k, 1, 2)), sqrt(self.model_dim)))
            multi_head_att_out = torch.cat([multi_head_att_out, torch.matmul(att_weights, v)], dim=-1)

        return self.multi_head_att_mapper(multi_head_att_out), residual_connection

class MedPoseConvLSTM(nn.Module):
    
    def __init__(self, num_layers, input_size, model_size, hidden_size, batch_first=False, lrnn_batch_norm=False, conv2d_req=True, conv1d_req=False, decoder_first=False):
        super(MedPoseConvLSTM, self).__init__()
        if conv2d_req:
            '''
            process image inputs with convolutions so that they can be 
            passed into an LSTM to capture dependencies over frame
            sequences
            '''
            if decoder_first:
                if lrnn_batch_norm:
                    self.conv2d = nn.Sequential(
                        nn.Conv2d(input_size, 32, kernel_size=3, padding=1, stride=2),
                        SyncBatchNorm(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(32, 68, kernel_size=3, padding=1, stride=2),
                        SyncBatchNorm(68),
                        nn.ReLU(),
                        #nn.MaxPool2d(kernel_size=2),
                        #nn.Conv2d(input_size * 4, input_size * 2, kernel_size=3, padding=1, stride=2),
                        #nn.BatchNorm2d(input_size * 2),
                        #nn.ReLU(),
                        #nn.MaxPool2d(kernel_size=2),
                        #nn.Conv2d(input_size * 2, input_size * 2, kernel_size=3, padding=1, stride=2),
                        #nn.BatchNorm2d(input_size * 2),
                        #nn.ReLU(),
                        #nn.Conv2d(input_size * 2, input_size, kernel_size=3, padding=1, stride=2),
                        #nn.BatchNorm2d(input_size),
                        #nn.ReLU()
                    )
                else:
                    self.conv2d = nn.Sequential(
                        nn.Conv2d(input_size, 32, kernel_size=3, padding=1, stride=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(32, 68, kernel_size=3, padding=1, stride=2),
                        nn.ReLU()
                    )
            else:
                if lrnn_batch_norm:
                    self.conv2d = nn.Sequential(
                                nn.Conv2d(input_size, input_size // 2, kernel_size=3, padding=1, stride=2),
                                nn.SyncBatchNorm(input_size // 2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(input_size // 2, input_size // 4, kernel_size=3, padding=1, stride=2),
                                nn.SyncBatchNorm(input_size // 4),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(input_size // 4, input_size // 2, kernel_size=3, padding=1, stride=2),
                                nn.SyncBatchNorm(input_size // 2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(input_size // 2, input_size // 2, kernel_size=3, padding=1, stride=2),
                                nn.SyncBatchNorm(input_size // 2),
                                nn.ReLU(),
                                nn.Conv2d(input_size // 2, input_size, kernel_size=3, padding=1, stride=2),
                                nn.SyncBatchNorm(input_size),
                                nn.ReLU()
                            )
                else:
                    self.conv2d = nn.Sequential(
                                nn.Conv2d(input_size, input_size // 2, kernel_size=3, padding=1, stride=2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(input_size // 2, input_size // 4, kernel_size=3, padding=1, stride=2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(input_size // 4, input_size // 2, kernel_size=3, padding=1, stride=2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(input_size // 2, input_size // 2, kernel_size=3, padding=1, stride=2),
                                nn.ReLU(),
                                nn.Conv2d(input_size // 2, input_size, kernel_size=3, padding=1, stride=2),
                                nn.ReLU()
                            )

        #if conv1d_req:
        #    '''
        #    in processing outputs of encoder layers instead of feature
        #    map inputs, 2d convolutions are substituted with 1d convolutions
        #    '''
            #pass
            # if lrnn_batch_norm:
            #     self.conv1d = nn.Sequential(
            #                 nn.Conv1d(model_size, model_size // 2, kernel_size=5, padding=1, stride=2),
            #                 nn.BatchNorm1d(model_size // 2),
            #                 nn.ReLU(),
            #                 nn.MaxPool1d(kernel_size=2),
            #                 nn.Conv1d(model_size // 2, model_size // 4, kernel_size=5, padding=1, stride=2),
            #                 nn.BatchNorm1d(model_size // 4),
            #                 nn.ReLU(),
            #                 nn.MaxPool1d(kernel_size=2),
            #                 nn.Conv1d(model_size // 4, model_size // 8, kernel_size=5, padding=1, stride=2),
            #                 nn.BatchNorm1d(model_size // 8),
            #                 nn.ReLU()
            #             )
            # else:
            #     self.conv1d = nn.Sequential(
            #                 nn.Conv1d(model_size, model_size // 2, kernel_size=5, padding=1, stride=2),
            #                 nn.ReLU(),
            #                 nn.MaxPool1d(kernel_size=2),
            #                 nn.Conv1d(model_size // 2, model_size // 4, kernel_size=5, padding=1, stride=2),
            #                 nn.ReLU(),
            #                 nn.MaxPool1d(kernel_size=2),
            #                 nn.Conv1d(model_size // 4, model_size // 8, kernel_size=5, padding=1, stride=2),
            #                 nn.ReLU()
            #             )
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
        self.model_dim = model_size

    def forward(self, x):
        # fixing bug resulting from allocation for variables/tensors
        x = x.contiguous()
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
            if len(x.shape) == 6:
                batch_size, num_frames, num_regions, num_keypoints, h, w = x.shape
                c_in = x.view(-1, num_keypoints, h, w)
            else:
                batch_size, num_frames, c, h, w = x.shape
                c_in = x.view(batch_size * num_frames, c, h, w)
            c_out = self.conv2d(c_in)
            if len(x.shape) == 6:
                c_out = c_out.view(batch_size, num_frames, num_regions, -1)
                r_in = c_out.view(batch_size, num_frames * num_regions, -1)
            else:
                r_in = c_out.view(batch_size, num_frames, -1)
            # ensure the input is of correct feature dim size
            if r_in.shape[2] > self.model_dim:
                r_in = F.max_pool1d(r_in, kernel_size=2)
        elif self.conv1d_req:
            x = x.permute(0, 1, 3, 2).contiguous()
            batch_size, num_frames, l, c = x.shape
            r_in = x.view(batch_size, num_frames * l, c)
            #c_in = c_in.cuda(next(self.conv1d.parameters()).get_device())
            #print(c_in.shape, "conv1d in")
            #c_out = self.conv1d(c_in)
            #print(c_out.shape, "conv1d out")
            #r_in = c_out.view(batch_size, num_frames, -1)
            #print(r_in.shape, "conv1d")
        else:
            r_in = x
        self.lstm.flatten_parameters()
        return self.lstm(r_in), r_in

class MedPoseLayerNorm(nn.Module):
    def __init__(self, feature_dim, eps=1e-6):
        super(MedPoseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod 
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

class BatchNorm2dWrapper(nn.BatchNorm2d):
    """
    wrapper around BatchNorm2d to allow for zero-size tensors
    """
    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)

class SyncBatchNorm(BatchNorm2dWrapper):
    def forward(self, input):
        world_size = 1 if not dist.is_available() or not dist.is_initialized() else dist.get_world_size()
        if world_size == 1 or not self.training:
            return super().forward(input)
        
        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"

        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias

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
