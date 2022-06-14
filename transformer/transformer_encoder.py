import tensorflow as tf

from tensorflow.keras import layers
from transformer.transformers_utils import point_wise_feed_forward_network, positional_encoding
from transformer.multiheadAttention import MultiHeadAttention


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1,  **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
        })
        return config

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(layers.Layer):
    def __init__(self, input_vocab_size, num_layers_top, num_layers_bot,
                 d_model, num_heads, dff, rate, maxlen, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers_top = num_layers_top
        self.num_layers_bot = num_layers_bot
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.rate = rate
        self.maxlen = maxlen

        # Define layers
        self.embedding = layers.Embedding(self.input_vocab_size, self.d_model)

        self.pos_encoding = positional_encoding(self.maxlen, self.d_model)

        self.dropout = layers.Dropout(self.rate)

        self.enc_layers_top = [EncoderLayer(d_model=self.d_model,
                                            num_heads=self.num_heads,
                                            dff=self.dff,
                                            rate=self.rate)
                               for _ in range(self.num_layers_top)]

        self.enc_layers_bot = [EncoderLayer(d_model=self.d_model,
                                            num_heads=self.num_heads,
                                            dff=self.dff,
                                            rate=self.rate)
                               for _ in range(self.num_layers_bot)]

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers_top": self.num_layers_top,
            "num_layers_bot": self.num_layers_bot,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "rate": self.rate,
            "maxlen": self.maxlen
        })
        return config

    def call(self, input, training, mask):

        seq_len = tf.shape(input)[1]

        # adding embedding and position encoding.
        x = self.embedding(input)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x1 = x_out = self.dropout(x, training=training)

        for i in range(self.num_layers_bot):
            x_out = self.enc_layers_bot[i](x_out, training, mask)

        x2 = (x1 + x_out)

        for i in range(self.num_layers_top):
            x2 = self.enc_layers_top[i](x2, training, mask)

        return x_out, x2  # (batch_size, input_seq_len, d_model)

if __name__ == '__main__':
    import tensorflow as tf

    # Test encoder layer
    sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048, rate=0.5)
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((16, 30, 512)), False, None)
    print(sample_encoder_layer_output.shape)

    # Test Encoder block
    sample_encoder = Encoder(input_vocab_size=6500, num_layers_top=2, num_layers_bot=6,
                             d_model=512, num_heads=8, dff=2048, rate=0.5)
    temp_input = tf.random.uniform((16, 30), dtype=tf.int64, minval=0, maxval=200)
    sample_encoder_output_bot, sample_encoder_output_top = sample_encoder(temp_input, training=True, mask=None)
    print(sample_encoder_output_bot.shape)  # (batch_size, input_seq_len, d_model)
    print(sample_encoder_output_top.shape)  # (batch_size, input_seq_len, d_model)