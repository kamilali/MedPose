import torch.nn as nn
import torch
from .mp_layers import MedPoseAttention, MedPoseConvLSTM

class MedPoseDecoder(nn.Module):

    def __init__(self, num_dec_layers=3, num_att_heads=4, num_lrnn_layers=3,
            model_dim=256, lrnn_hidden_dim=256, ff_hidden_dim=1024,
            roi_map_dim=7, lrnn_window_size=3, num_keypoints=17, lrnn_batch_norm=False, use_lrnn=True, dec_history=[], gpus=None, device=None):
        super(MedPoseDecoder, self).__init__()
        '''
        store number of decoder layers and a dictionary containing
        history of outputs per encoder layer for recurrence
        Also create module list for all sub-layers to keep track of 
        them for each decoder layer
        '''
        self.num_dec_layers = num_dec_layers
        self.dec_history = dec_history

        if use_lrnn:
            self.local_rnns = nn.ModuleList()
            self.lrnn_layer_norms = nn.ModuleList()
            self.lrnn_window_size = lrnn_window_size

        self.self_atts = nn.ModuleList()
        self.self_att_layer_norms = nn.ModuleList()

        self.enc_dec_atts = nn.ModuleList()
        self.enc_dec_att_layer_norms = nn.ModuleList()

        self.ffs = nn.ModuleList()
        self.ff_layer_norms = nn.ModuleList()
        '''
        initialize stack layers
        '''
        for dec_layer in range(self.num_dec_layers):
            '''
            LocalRNN for capturing local structures to attend to
            However, since we are dealing with images, we modify the
            R-Transformer slightly by using a ConvLSTM instead of a 
            LSTM as the LocalRNN module
            '''
            if use_lrnn:
                local_rnn = MedPoseConvLSTM(
                        num_layers=num_lrnn_layers, 
                        input_size=num_keypoints,
                        model_size=model_dim,
                        hidden_size=lrnn_hidden_dim,
                        batch_first=True,
                        lrnn_batch_norm=lrnn_batch_norm,
                        conv2d_req=(dec_layer == 0),
                        conv1d_req=(dec_layer != 0),
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

    def forward(self, enc_out, poses=None, initial_frame=True):
        '''
        stack decoders based on number of decoder layers specified
        '''
        for dec_layer in range(self.num_dec_layers):
            #input("entered decoder layer " + str(dec_layer))
            if initial_frame:
                self.dec_history[enc_out.get_device() - 1].reset()
                curr_dec_hist = self.dec_history[enc_out.get_device() - 1]
                '''
                use encoder as query and context for first pose detection
                (skip the first two layers of the decoder since those
                require prior pose estimations)
                '''
                eda_out, residual_connection = self.enc_dec_atts[dec_layer](enc_out, enc_out)
                #eda_out, residual_connection = data_parallel(self.enc_dec_atts[dec_layer], (enc_out, enc_out), self.gpus, self.device)
                '''
                layer normalization +  residual connection
                '''
                eda_out = self.enc_dec_att_layer_norms[dec_layer](eda_out + residual_connection)
                #eda_out = data_parallel(self.enc_dec_att_layer_norms[dec_layer], eda_out + residual_connection, self.gpus, self.device)
                '''
                transform features non-linearly through feed forward
                module
                '''
                dec_out = self.ffs[dec_layer](eda_out)
                #dec_out = data_parallel(self.ffs[dec_layer], eda_out, self.gpus, self.device)
                '''
                layer normalization + residual connection
                '''
                dec_out = self.ff_layer_norms[dec_layer](dec_out + eda_out)
                #dec_out = data_parallel(self.ff_layer_norms[dec_layer], dec_out + eda_out, self.gpus, self.device)
            else:
                curr_dec_hist = self.dec_history[enc_out.get_device() - 1]
                if dec_layer == 0:
                    poses = poses[:, -self.lrnn_window_size:]
                    #poses = poses.permute(0, 1, 3, 2)
                    (context, _), residual_connection = self.local_rnns[dec_layer](poses)
                    #(context, _), residual_connection = data_parallel(self.local_rnns[dec_layer], poses, self.gpus, self.device)
                else:
                    dec_in = torch.stack(curr_dec_hist.get_history(dec_layer), dim=1)
                    dec_in = dec_in.permute(0, 1, 3, 2)
                    (context, _), residual_connection = self.local_rnns[dec_layer](dec_in)
                    #(context, _), residual_connection = data_parallel(self.local_rnns[dec_layer], dec_in, self.gpus, self.device)
                '''
                layer normalization + residual connection (comes from output
                of conv layers prior to forward pass through lstm)
                '''
                context = self.lrnn_layer_norms[dec_layer](context + residual_connection)
                #context = data_parallel(self.lrnn_layer_norms[dec_layer], context + residual_connection, self.gpus, self.device)
                '''
                use the last hidden layer as the hidden representation
                '''
                context = context[:, context.shape[1] - 1]
                context = context.unsqueeze(dim=1)
                '''
                formulate query and past_context for attention
                mechanism
                '''
                query = context
                past_context = curr_dec_hist.get_lrnn_history(dec_layer)
                if past_context is None:
                    past_context = context
                '''
                attend to local structures using hidden representation
                as query and context for self attention
                '''
                sa_out, residual_connection = self.self_atts[dec_layer](query, past_context)
                #sa_out, residual_connection = data_parallel(self.self_atts[dec_layer], (query, past_context), self.gpus, self.device)
                '''
                layer normalization + residual connection
                '''
                sa_out = self.self_att_layer_norms[dec_layer](sa_out + residual_connection)
                #sa_out = data_parallel(self.self_att_layer_norms[dec_layer], sa_out + residual_connection, self.gpus, self.device)
                '''
                add hidden rnn output to history for attending over past
                local structures
                '''
                # if self.hr_hist_device[dec_layer] is None:
                #     self.hr_hist_device[dec_layer] = context.get_device()
                #     self.hidden_rnn_hist[dec_layer] = context
                # else:
                #     self.hidden_rnn_hist[dec_layer] = torch.cat((self.hidden_rnn_hist[dec_layer], context.to(self.hr_hist_device[dec_layer])), dim=1)
                if curr_dec_hist.get_lrnn_history(dec_layer) is None:
                    #curr_dec_hist.set_lrnn_history_device(dec_layer, context.get_device())
                    curr_dec_hist.set_lrnn_history(dec_layer, context)
                else:
                    #curr_dec_hist.set_lrnn_history(dec_layer, torch.cat((curr_dec_hist.get_lrnn_history(dec_layer), context.to(curr_dec_hist.get_lrnn_history_device(dec_layer))), dim=1))
                    curr_dec_hist.set_lrnn_history(dec_layer, torch.cat((curr_dec_hist.get_lrnn_history(dec_layer), context), dim=1))
                '''
                attend to local structures using output of prior
                attention layer as queries and the corresponding
                encoder output as the context
                '''
                eda_out, residual_connection = self.enc_dec_atts[dec_layer](sa_out, enc_out)
                #eda_out, residual_connection = data_parallel(self.enc_dec_atts[dec_layer], (sa_out, enc_out), self.gpus, self.device)
                '''
                layer normalization + residual connection
                '''
                eda_out = self.enc_dec_att_layer_norms[dec_layer](eda_out + enc_out)
                #eda_out = data_parallel(self.enc_dec_att_layer_norms[dec_layer], eda_out + enc_out, self.gpus, self.device)
                '''
                transform features non-linearly through feed forward
                module
                '''
                dec_out = self.ffs[dec_layer](eda_out)
                #dec_out = data_parallel(self.ffs[dec_layer], eda_out, self.gpus, self.device)
                '''
                layer normalization + residual connection
                '''
                dec_out = self.ff_layer_norms[dec_layer](dec_out + eda_out)
                #dec_out = data_parallel(self.ff_layer_norms[dec_layer], dec_out + eda_out, self.gpus, self.device)
            '''
            store current output in decoder history for next layer and, 
            if the next layer history is full, shift history to
            make room for new output (current output is input for
            next layer)
            '''
            if dec_layer < (self.num_dec_layers - 1):
                # if self.hist_device[dec_layer + 1] is None:
                #     self.hist_device[dec_layer + 1] = dec_out.get_device()
                # if len(self.hist[dec_layer + 1]) == self.lrnn_window_size:
                #     self.hist[dec_layer + 1] = self.hist[dec_layer + 1][1:]
                # self.hist[dec_layer + 1].append(dec_out.to(self.hist_device[dec_layer + 1]))
                #if curr_dec_hist.get_history_device(dec_layer + 1) is None:
                #    curr_dec_hist.set_history_device(dec_layer + 1, dec_out.get_device())
                if curr_dec_hist.get_history_size(dec_layer + 1) == self.lrnn_window_size:
                    curr_dec_hist.set_history(dec_layer + 1, curr_dec_hist.get_history(dec_layer + 1)[1:])
                #curr_dec_hist.append_history(dec_layer + 1, dec_out.to(curr_dec_hist.get_history_device(dec_layer + 1)))
                curr_dec_hist.append_history(dec_layer + 1, dec_out)
            
        return dec_out