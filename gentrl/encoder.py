#zz import torch
#zz from torch import nn
import tensorflow as tf
from tensorflow.keras import layers
from gentrl.tokenizer import encode, get_vocab_size

class RNNEncoder(layers.Layer):
    def __init__(self, hidden_size=256, num_layers=2, latent_size=50,
                 bidirectional=False):
        super(RNNEncoder, self).__init__()

        """
        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 2 * latent_size))
        """

        self.embs = layers.Embedding(input_dim=get_vocab_size(), output_dim=hidden_size)
        self.rnn = tf.keras.Sequential()
        for i in range(num_layers):
            if bidirectional == True:
                self.rnn.add(layers.Bidirectional(layer=layers.GRU(units=hidden_size, return_sequences=True)))
            else:
                self.rnn.add(layer=layers.GRU(units=hidden_size, return_sequences=True))
                self.final_mlp = tf.keras.Sequential()

        self.final_mlp.add(layers.Dense(hidden_size))
        self.final_mlp.add(layers.LeakyReLU())
        self.final_mlp.add(layers.Dense(2 * latent_size))

    def encode(self, sm_list):
        """
        Maps smiles onto a latent space
        """
        
        """
        tokens, lens = encode(sm_list)
        to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)

        outputs = self.rnn(self.embs(to_feed))[0]
        outputs = outputs[lens, torch.arange(len(lens))]

        return self.final_mlp(outputs)
        """

        tokens, lens = encode(sm_list.numpy().astype(str))
        to_feed = tokens
        # print(to_feed)
        outputs = self.rnn(self.embs(to_feed))
        idx = [list(a) for a in zip(list(tf.range(len(lens)).numpy()), lens)]
        outputs = tf.gather_nd(outputs, idx)
        
        return self.final_mlp(outputs)
