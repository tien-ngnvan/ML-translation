import argparse
import sys
import time

from load_data import load_vectorization_from_disk, process_data, make_dataset
from transformer.transformer_model import Transformers
from bleu_score import bleu_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path_test_en', type=str, default='./envi-nlp/test.en',
                        help='test.en')
    parser.add_argument('--path_test_vi', type=str, default='./envi-nlp/test.vi',
                        help='test.vi')
    parser.add_argument('--path_source_token', type=str, default=r'./source_vectorization_layer.pkl',
                        help='Tokenizer source')
    parser.add_argument('--path_target_token', type=str, default=r'./target_vectorization_layer.pkl',
                        help='Tokenizer target')
    parser.add_argument('--path_ckpt', type=str, default=r'./ckpt/transformer.h5',
                        help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    # load token
    source_token = load_vectorization_from_disk(args.path_source_token)
    target_token = load_vectorization_from_disk(args.path_target_token)
    vocab_input = source_token.vocabulary_size()
    vocab_target = target_token.vocabulary_size()

    # Read_data
    with open(args.path_test_en, 'r', encoding='utf-8') as f:
        lines_en_test = f.readlines()
    with open(args.path_test_vi, 'r', encoding='utf-8') as f:
        lines_vi_test = f.readlines()

    # load model
    transformer = Transformers(source_vectorization=source_token,
                               target_vectorization=target_token,
                               vocab_input=vocab_input,
                               vocab_target=vocab_target,
                               training=False)
    transformer.build_model()
    transformer.load_model(model_path=args.path_ckpt)

    # Predict 
    predict_ls = []
    start = time.time()
    for text in lines_en_test:
        result = transformer.predict(process_data(text))
        predict_ls.append(result)
    print('Predict time: ', start - time.time())

    # write file
    with open('predict.txt', 'w') as file:
        file.write(predict_ls)
    
    # bleu score
    path_refs = ['predict.txt']
    path_hyp = '/content/drive/MyDrive/translation/envi-nlp/test.vi'

    result = bleu_score(path_refs, path_hyp, n_lines=None)
    for i in range(4):
        print('Blue score in {}_gram: {}'.format(i+1, result[i]) )




