import tensorflow as tf
import numpy as np

from transformer.transformer_encoder import EncoderLayer, Encoder
from transformer.transformer_decoder import DecoderLayer, Decoder
from transformer.multiheadAttention import MultiHeadAttention
from transformer.transformers_utils import *
from losses import scce_with_ls


class Transformers():
    def __init__(self, source_token, target_token, vocab_input, vocab_target,
                 n_layers_top=2, n_layers_bot=6, d_model=512, n_heads=8, maxlen=256,
                 dff=2048, drop_rate=0.1, model_path=None, training=True):

        self.source_token = source_token
        self.target_token = target_token
        self.num_layers_top = n_layers_top
        self.num_layers_bot = n_layers_bot
        self.d_model = d_model
        self.n_heads = n_heads
        self.dff = dff
        self.vocab_input = vocab_input
        self.vocab_target = vocab_target
        self.drop_rate = drop_rate
        self.maxlen = maxlen
        self.load_model(model_path)
        self.training = training

    def load_model(self, model_path):
        if model_path is None:
            return
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'MultiHeadAttention': MultiHeadAttention,
                'EncoderLayer': EncoderLayer,
                'Encoder': Encoder,
                'DecoderLayer': DecoderLayer,
                'Decoder': Decoder,
                'scce_with_ls': scce_with_ls,
            })

    def build_model(self, *args, **kwargs):

        ################ create input model ################
        inp = tf.keras.Input(shape=(None,), dtype="int64", name="inp_encoder")
        tar = tf.keras.Input(shape=(None,), dtype="int64", name="inp_decoder")

        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        ###################### encoder ######################
        encoder = Encoder(num_layers_top=self.num_layers_top,
                          num_layers_bot=self.num_layers_bot, maxlen = self.maxlen,
                          d_model=self.d_model, num_heads=self.n_heads, dff=self.dff,
                          input_vocab_size=self.vocab_input, rate=self.drop_rate)

        enc_output_bot, enc_output_top = encoder(inp, self.training, padding_mask)

        ###################### decoder ######################
        decoder = Decoder(num_layers_top=self.num_layers_top,
                          num_layers_bot=self.num_layers_bot, maxlen = self.maxlen,
                          d_model=self.d_model, num_heads=self.n_heads, dff=self.dff,
                          target_vocab_size=self.vocab_target, rate=self.drop_rate)

        dec_output, attention_weights = decoder(tar, enc_output_bot, enc_output_top,
                                                self.training, look_ahead_mask, padding_mask)

        decoder_outputs = tf.keras.layers.Dense(self.vocab_target)(dec_output)

        #################### build model ####################
        self.model = tf.keras.Model([inp, tar], decoder_outputs)
        self.model.compile(*args, **kwargs)

    def summary(self, *args, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            raise TypeError('Please build the model or load a pre-trained model first')

        self.model.summary()

    def fit(self, *args, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            raise TypeError('Please build the model or load a pre-trained model first')

        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            raise TypeError('Please build the model or load a pre-trained model first')

        rs = self.model.evaluate(*args, **kwargs)
        print(rs)

    def predict(self, input_sentence):
        if not hasattr(self, 'model') or self.model is None:
            raise TypeError('Please load a pre-trained model first')

        self.original_index_lookup = dict(zip(range(len(self.target_token.get_vocab())),
                                              self.target_token.get_vocab()))

        # input encoder
        input_word = input_sentence.split()
        
        inp1 = self.source_token.encode_plus(input_sentence, max_length=self.maxlen, truncation=True,
                                   add_special_tokens=False, padding='max_length', return_tensors='tf')
        tokenized_input_sentence = inp1['input_ids']

        decoded_sentence = "start"
        for i in range(0, self.maxlen):
            out_token = self.source_token.encode_plus(decoded_sentence, max_length=self.maxlen, truncation=True,
                                   add_special_tokens=False, padding='max_length', return_tensors='tf')
            inp2 = out_token['input_ids']
        
            tokenized_target_sentence = inp2[:,:-1]
            # predict
            predictions = self.model(
                [tokenized_input_sentence, tokenized_target_sentence])
            
            sampled_token_index = np.argmax(predictions[0, i, :])
            
            if sampled_token_index == 0:
                break
            else:
                sampled_token = self.original_index_lookup[sampled_token_index]
             
                if sampled_token != 'end':
                    decoded_sentence += " " + str(sampled_token)

                if (sampled_token == 'end') or len(decoded_sentence.split()) >= 50:
                    break

        return decoded_sentence.replace('start', '')

    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask