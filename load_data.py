import tensorflow as tf
import re
import pickle
import numpy as np


def process_data(data):
    text = data.strip().lower()
    text = re.sub(r'([!.?])', r' \1', text)
    text = re.sub(r"\s+", r' ', text)
    text = re.sub("&apos;", "'", text)
    text = re.sub("&quot;", "", text)
    text = re.sub("&#", "", text) 

    return text

def format_dataset(inp_text, inp_tgt):    
    return ({
        'inp_encoder': inp_text,
        'inp_decoder': inp_tgt[:, :-1]
    }, inp_tgt[:, 1:])

def get_token(text, tokenizer, maxlen):
    tokens = tokenizer.encode_plus(text, max_length=maxlen, truncation=True,
                                   add_special_tokens=False, padding='max_length', 
                                   return_tensors='tf')
    return np.asarray(tokens['input_ids'][0])

def make_dataset(pairs, source_token, target_token, batch_size, maxlen):
    enc_texts, dec_texts = pairs

    enc_texts = [get_token(text, source_token, maxlen) for text in enc_texts]
    dec_texts = [get_token(text, target_token, maxlen) for text in dec_texts]

    dataset = tf.data.Dataset.from_tensor_slices((enc_texts, dec_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: format_dataset(x, y), num_parallel_calls= tf.data.AUTOTUNE)

    return dataset.shuffle(2048).prefetch(tf.data.AUTOTUNE).cache()