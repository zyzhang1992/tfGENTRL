#import torch
#import torch.nn as nn
#import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras import layers
from gentrl.tokenizer import get_vocab_size, encode, decode


class DilConvDecoder(layers.Layer):
    '''
    Class for autoregressive model that works in WaveNet manner.
        It make conditinioning on previosly sampled tokens by running
        stack of dilation convolution on them.
    '''
    def __init__(self, latent_input_size, token_weights=None,
                 split_len=50, num_dilated_layers=7, num_channels=128):
        r'''
        Args:
            latent_input_size: int, size of latent code used in VAE-like models
            token_weights: Tensor of shape [num_tokens], where i-th element
                    contains the weight of i-th token. If None, then all
                    tokens has the same weight.
            split_len: int, maximum length of token sequence
            num_dilated_layers: int, how much dilated layers is in stack
            num_channels: int, num channels in convolutional layers
        '''
        super(DilConvDecoder, self).__init__()
        self.vocab_size = get_vocab_size()
        self.latent_input_size = latent_input_size
        self.split_len = split_len
        self.num_dilated_layers = num_dilated_layers
        self.num_channels = num_channels
        self.token_weights = token_weights
        self.eos = 2

        cur_dil = 1
        self.dil_conv_layers = []
        for i in range(num_dilated_layers):
            self.dil_conv_layers.append(
                    DilConv1dWithGLU(num_channels, cur_dil))
            cur_dil *= 2

        #zz self.latent_fc = nn.Linear(latent_input_size, num_channels)
        #zz self.input_embeddings = nn.Embedding(self.vocab_size,
        #zz                                      num_channels)
        #zz self.logits_1x1_layer = nn.Conv1d(num_channels,
        #zz                                   self.vocab_size,
        #zz                                   kernel_size=1)
        self.latent_fc = layers.Dense(num_channels)
        self.input_embeddings = layers.Embedding(self.vocab_size,
                                             num_channels)
        self.logits_1x1_layer = layers.Conv1D(filters=self.vocab_size,
                                              kernel_size=1,
                                              input_shape=[latent_input_size, num_channels])

        #zz cur_parameters = []
        #zz for layer in [self.input_embeddings, self.logits_1x1_layer,
        #zz               self.latent_fc] + self.dil_conv_layers:
        #zz     cur_parameters += list(layer.parameters())

        #zz self.parameters = nn.ParameterList(cur_parameters)

    def get_logits(self, input_tensor, z, sampling=False):
        '''
        Computing logits for each token input_tensor by given latent code

        [WORKS ONLY IN TEACHER-FORCING MODE]

        Args:
            input_tensor: Tensor of shape [batch_size, max_seq_len]
            z: Tensor of shape [batch_size, lat_code_size]
        '''

        #zz input_embedded = self.input_embeddings(input_tensor).transpose(1, 2)
        input_embedded = self.input_embeddings(input_tensor)
        latent_embedded = self.latent_fc(z)

        #zz x = input_embedded + latent_embedded.unsqueeze(-1)
        x = input_embedded + tf.expand_dims(input=latent_embedded, axis=1)

        for dil_conv_layer in self.dil_conv_layers:
            x = dil_conv_layer(x, sampling=sampling)

        #zz x = self.logits_1x1_layer(x).transpose(1, 2)
        x = self.logits_1x1_layer(x)

        #zz return F.log_softmax(x, dim=-1)
        return tf.nn.log_softmax(x, axis=-1)

    def get_log_prob(self, x, z):
        '''
        Getting logits of SMILES sequences
        Args:
            x: tensor of shape [batch_size, seq_size] with tokens
            z: tensor of shape [batch_size, lat_size] with latents
        Returns:
            logits: tensor of shape [batch_size, seq_size]
        '''
        #zz seq_logits = torch.gather(self.get_logits(x, z)[:, :-1, :],
        #zz                           2, x[:, 1:].long().unsqueeze(-1))
        idxzip = [[list(a) for a in zip(range(x.shape[1]), list(x[b, 1:].numpy()))] 
                  for b in range(x.shape[0])]
        idx = [[[j] + idxzip[j][i] for i in range(x.shape[1] - 1)] for j in range(x.shape[0])]
        idx = tf.convert_to_tensor(idx)
        seq_logits = tf.expand_dims(tf.gather_nd(self.get_logits(x, z)[:, :-1, :], idx), -1)
        
        return seq_logits[:, :, 0]

    def forward(self, x, z):
        '''
        Getting logits of SMILES sequences
        Args:
            x: tensor of shape [batch_size, seq_size] with tokens
            z: tensor of shape [batch_size, lat_size] with latents
        Returns:
            logits: tensor of shape [batch_size, seq_size]
            None: since dilconv decoder doesn't have hidden state unlike RNN
        '''
        return self.get_log_prob(x, z), None

    def weighted_forward(self, sm_list, z):
        '''
        '''
        #zz x = encode(sm_list)[0].to(
        #zz     self.input_embeddings.weight.data.device
        #zz )
        x = encode(sm_list.numpy().astype(str))[0]

        seq_logits = self.get_log_prob(x, z)

        if self.token_weights is not None:
            #zz w = self.token_weights[x[:, 1:].long().contiguous().view(-1)]
            #zz w = w.view_as(seq_logits)
            w = self.token_weights[tf.reshape(tf.cast(x[:, 1:], tf.int64), shape=-1)]
            w = tf.reshape(w, shape=seq_logits.shape)
            seq_logits = seq_logits * w

        #zz non_eof = (x != self.eos)[:, :-1].float()
        #zz ans_logits = (seq_logits * non_eof).sum(dim=-1)
        #zz ans_logits /= non_eof.sum(dim=-1)
        non_eof = tf.cast((x != self.eos)[:, :-1], tf.float32)
        ans_logits = tf.reduce_sum((seq_logits * non_eof), axis=-1)
        ans_logits /= tf.reduce_sum(non_eof, axis=-1)

        return ans_logits

    def sample(self, max_len, latents, argmax=True):
        ''' Sample SMILES for given latents

        Args:
            latents: tensor of shape [n_batch, n_features]

        Returns:
            logits: tensor of shape [batch_size, seq_size], logits of tokens
            tokens: tensor of shape [batch_size, seq_size], sampled token
            None: since dilconv decoder doesn't have hidden state unlike RNN

        '''

        # clearing buffers
        for dil_conv_layer in self.dil_conv_layers:
            dil_conv_layer.clear_buffer()

        num_objects = latents.shape[0]

        ans_seqs = [[1] for _ in range(num_objects)]
        ans_logits = []

        cur_tokens = torch.tensor(ans_seqs, device=latents.device).long()

        for s in range(max_len):
            logits = self.get_logits(cur_tokens, latents, sampling=True)
            logits = logits.detach()
            logits = torch.log_softmax(logits[:, 0, :], dim=-1)
            ans_logits.append(logits.unsqueeze(0))

            if argmax:
                cur_tokens = torch.max(logits, dim=-1)[1].unsqueeze(-1)
            else:
                cur_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)

            det_tokens = cur_tokens.cpu().detach().tolist()
            ans_seqs = [a + b for a, b in zip(ans_seqs, det_tokens)]

        # clearing buffers
        for dil_conv_layer in self.dil_conv_layers:
            dil_conv_layer.clear_buffer()

        ans_logits = torch.cat(ans_logits, dim=0)
        ans_seqs = torch.tensor(ans_seqs)[:, 1:]
        return decode(ans_seqs)


#zz class DilConv1dWithGLU(nn.Module):
class DilConv1dWithGLU(layers.Layer):
    def __init__(self, num_channels, dilation, lenght=100,
                 kernel_size=2, activation=tf.nn.leaky_relu,
                 residual_connection=True, dropout=0.2):

        super(DilConv1dWithGLU, self).__init__()

        self.dilation = dilation

        #zz self.start_ln = nn.LayerNorm(num_channels)
        #zz self.start_conv1x1 = nn.Conv1d(num_channels, num_channels,
        #zz                                kernel_size=1)
        self.start_ln = layers.LayerNormalization(axis=-1)
        self.start_conv1x1 = layers.Conv1D(filters=num_channels, 
                                           kernel_size=1)

        #zz self.dilconv_ln = nn.LayerNorm(num_channels)
        #zz self.dilated_conv = nn.Conv1d(num_channels, num_channels,
        #zz                               dilation=dilation,
        #zz                               kernel_size=kernel_size,
        #zz                               padding=dilation)
        self.dilconv_ln = layers.LayerNormalization(axis=-1)
        self.dilated_conv = layers.Conv1D(filters=num_channels, 
                                          dilation_rate=dilation, 
                                          kernel_size=kernel_size, 
                                          padding='causal')

        #zz self.gate_ln = nn.LayerNorm(num_channels)
        #zz self.end_conv1x1 = nn.Conv1d(num_channels, num_channels,
        #zz                              kernel_size=1)
        #zz self.gated_conv1x1 = nn.Conv1d(num_channels, num_channels,
        self.gate_ln = layers.LayerNormalization(axis=-1)
        self.end_conv1x1 = layers.Conv1D(filters=num_channels, 
                                         kernel_size=1)
        self.gated_conv1x1 = layers.Conv1D(filters=num_channels, 
                                           kernel_size=1)

        self.activation = activation

        self.buffer = None

        self.residual_connection = residual_connection

    def clear_buffer(self):
        self.buffer = None

    def forward(self, x_inp, sampling=False):
        # applying 1x1 convolution
        #zz x = self.start_ln(x_inp.transpose(1, 2)).transpose(1, 2)
        x = self.start_ln(x_inp)
        x = self.activation(x)
        x = self.start_conv1x1(x)

        # applying dilated convolution
        # if in sampling mode
        #zz x = self.dilconv_ln(x.transpose(1, 2)).transpose(1, 2)
        x = self.dilconv_ln(x)
        x = self.activation(x)
        if sampling:
            if self.buffer is None:
                self.buffer = x
            else:
                #zz pre_buffer = torch.cat([self.buffer, x], dim=2)
                pre_buffer = tf.concat([self.buffer, x], axis=2)
                self.buffer = pre_buffer[:, :, -(self.dilation + 1):]

            if self.buffer.shape[2] == self.dilation + 1:
                x = self.buffer
            else:
                #zz x = torch.cat([torch.zeros(self.buffer.shape[0],
                #zz                           self.buffer.shape[1],
                #zz                           self.dilation + 1
                #zz                           - self.buffer.shape[2],
                #zz                           device=x_inp.device), self.buffer],
                #zz               dim=2)
                x = tf.concat([tf.zeros([self.buffer.shape[0],
                                         self.buffer.shape[1],
                                         self.dilation + 1
                                         - self.buffer.shape[2]]), self.buffer],
                              axis=2)

            x = self.dilated_conv(x)[:, :, self.dilation:]
            x = x[:, :, :x_inp.shape[-1]]
        else:
            x = self.dilated_conv(x)[:, :, :x_inp.shape[-1]]

        # applying gated linear unit
        #zz x = self.gate_ln(x.transpose(1, 2)).transpose(1, 2)
        x = self.gate_ln(x)
        x = self.activation(x)
        #zz x = self.end_conv1x1(x) * torch.sigmoid(self.gated_conv1x1(x))
        x = self.end_conv1x1(x) * tf.sigmoid(self.gated_conv1x1(x))

        # if residual connection
        if self.residual_connection:
            x = x + x_inp

        return x
