import tensorflow as tf

from tensorflow.keras import layers
from transformer.transformers_utils import positional_encoding, point_wise_feed_forward_network
from transformer.multiheadAttention import MultiHeadAttention


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        # Define layers
        self.mha1 = MultiHeadAttention(d_model=self.d_model,
                                       num_heads=self.num_heads)

        self.mha2 = MultiHeadAttention(d_model=self.d_model,
                                       num_heads=self.num_heads)

        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)
        self.dropout3 = layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
        })
        return config

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch, inp_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(layers.Layer):
    def __init__(self, target_vocab_size, num_layers_top, num_layers_bot,
                 d_model, num_heads, dff, maxlen, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model
        self.num_layers_top = num_layers_top
        self.num_layers_bot = num_layers_bot
        self.dff = dff
        self.rate = rate
        self.num_heads = num_heads
        self.maxlen = maxlen

        # Define layers
        self.embedding = layers.Embedding(self.target_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(self.maxlen, self.d_model)

        self.dec_layers_bot = [DecoderLayer(d_model=self.d_model,
                                            num_heads=self.num_heads,
                                            dff=self.dff,
                                            rate=self.rate)
                               for _ in range(self.num_layers_bot)]

        self.dec_layers_top = [DecoderLayer(d_model=self.d_model,
                                            num_heads=self.num_heads,
                                            dff=self.dff,
                                            rate=self.rate)
                               for _ in range(self.num_layers_top)]

        self.dropout = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_vocab_size": self.target_vocab_size,
            "num_layers_top": self.num_layers_top,
            "num_layers_bot": self.num_layers_bot,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
            'maxlen': self.maxlen
        })
        return config

    def call(self, x, enc_output_bot, enc_output_top, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x1 = x = self.dropout(x, training=training)

        for i in range(self.num_layers_bot):
            x, block1_bot, block2_bot = self.dec_layers_bot[i](x, enc_output_bot, training,
                                                               look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer_bot{i + 1}_block1'] = block1_bot
            attention_weights[f'decoder_layer_bot{i + 1}_block2'] = block2_bot

        x2 = (x1 + x)

        for i in range(self.num_layers_top):
            x2, block1_top, block2_top = self.dec_layers_top[i](x2, enc_output_top, training,
                                                                look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer_top{i + 1}_block1'] = block1_top
            attention_weights[f'decoder_layer_top{i + 1}_block2'] = block2_top

        # x2.shape == (batch_size, target_seq_len, d_model)
        return x2, attention_weights