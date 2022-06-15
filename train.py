import tensorflow as tf
import argparse
import sys

from transformer.transformer_model import Transformers
from losses import scce_with_ls
from load_data import process_data, make_dataset, vncore_tokenizer
from transformers import AutoTokenizer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_model": tf.cast(self.d_model, tf.float32),
            "warmup_steps": self.warmup_steps,
        })
        return config

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def train(args):

    # Read_data
    with open(args.path_train_en, 'r', encoding='utf-8') as f:
        lines_en_train = f.readlines()
    with open(args.path_train_vi, 'r', encoding='utf-8') as f:
        lines_vi_train = f.readlines()
    lines_en_train = [process_data(text) for text in lines_en_train]
    lines_vi_train = [process_data(text) for text in lines_vi_train]
    lines_vi_train = [vncore_tokenizer(text) for text in lines_vi_train]
    train_pairs = [lines_en_train, lines_vi_train]

    with open(args.path_val_en, 'r', encoding='utf-8') as f:
        lines_en_dev = f.readlines()
    with open(args.path_val_vi, 'r', encoding='utf-8') as f:
        lines_vi_dev = f.readlines()
    lines_en_dev = [process_data(text) for text in lines_en_dev]
    lines_vi_dev = [process_data(text) for text in lines_vi_dev]
    lines_vi_dev = [vncore_tokenizer(text) for text in lines_vi_dev]
    val_pairs = [lines_en_dev, lines_vi_dev]

    # load Tokenizer
    VietTokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    EngTokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = make_dataset(train_pairs, EngTokenizer, VietTokenizer, args.batch_size, args.maxlen)
    val_ds = make_dataset(val_pairs, EngTokenizer, VietTokenizer, args.batch_size, args.maxlen)
    print('Load token done!!! \n')
    vocab_input = EngTokenizer.vocab_size 
    vocab_target = VietTokenizer.vocab_size 

                           
    # build model
    transformer = Transformers(source_token=EngTokenizer, target_token=VietTokenizer,
                               vocab_input=vocab_input, vocab_target=vocab_target, n_layers_top=args.top_layers,
                               n_layers_bot=args.bot_layers, d_model=args.d_model, n_heads=args.num_heads,
                               maxlen=args.maxlen, dff=args.dense, drop_rate=args.dropout)

    learning_rate = CustomSchedule(args.d_model, args.warmup_step)
    optimizer = tf.keras.optimizers.Adam(learning_rate(1. * args.train_step), beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-8, clipnorm=0.1)

    callback = [tf.keras.callbacks.ModelCheckpoint(filepath=args.path_ckpt, monitor='val_loss',
                                                   mode='min', save_best_only=True)]

    # calculated step training
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()

    epoch = int(args.train_step // steps_per_epoch)
    transformer.build_model(optimizer=optimizer, loss=scce_with_ls, 
                            metrics=['sparse_categorical_accuracy'])
    
    print('-------------------- SUMMARY -------------------')
    transformer.summary()
    print('step per epochs: ', steps_per_epoch)
    print('Training step: ', args.train_step)
    print('Vocab input: ', vocab_input)
    print('Vocab target: ', vocab_target)
    

    transformer.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks=callback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--path_train_en', type=str, default='./envi-nlp/train.en',
                        help='train.en')
    parser.add_argument('--path_train_vi', type=str, default='./envi-nlp/train.vi',
                        help='train.vi')
    parser.add_argument('--path_val_en', type=str, default='./envi-nlp/dev.en',
                        help='val.en')
    parser.add_argument('--path_val_vi', type=str, default='./envi-nlp/dev.vi',
                        help='val.vi')
    parser.add_argument('--path_source_token', type=str, default=r'./source_vectorization_layer.pkl',
                        help='Tokenizer source')
    parser.add_argument('--path_target_token', type=str, default=r'./target_vectorization_layer.pkl',
                        help='Tokenizer target')
    parser.add_argument('--path_ckpt', type=str, default=r'./ckpt/transformer.h5',
                        help='checkpoint path')

    # model
    parser.add_argument('--d_model', type=int, default=512, help='d_model')
    parser.add_argument('--dense', type=int, default=2048, help='dense')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout layer')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_heads', type=int, default=8, help='num_layer multihead')
    parser.add_argument('--top_layers', type=int, default=2, help='top layers')
    parser.add_argument('--bot_layers', type=int, default=6, help='bottom layers')
    parser.add_argument('--maxlen', type=int, default=128, help='sequence token')

    # optimizer
    parser.add_argument('--train_step', type=int, default=40000, help='training step')
    parser.add_argument('--warmup_step', type=int, default=4000, help='warmup step')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    # training
    train(args)


