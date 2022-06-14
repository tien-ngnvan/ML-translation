from load_data import load_vectorization_from_disk, process_data
from transformer.transformer_model import Transformers


def  load_model(path):
    # load token
    source_token = load_vectorization_from_disk('/content/drive/MyDrive/translation/envi-nlp/source_vectorization_layer.pkl')
    target_token = load_vectorization_from_disk('/content/drive/MyDrive/translation/envi-nlp/target_vectorization_layer.pkl')
    vocab_input = source_token.vocabulary_size()
    vocab_target = target_token.vocabulary_size()

    # load model
    transformer = Transformers(source_vectorization=source_token,
                               target_vectorization=target_token,
                               vocab_input=vocab_input,
                               vocab_target=vocab_target,
                               training=False)
    transformer.build_model()
    transformer.load_model(path)

    return transformer

if __name__ == '__main__':
    path_ckpt = '/content/drive/MyDrive/translation/ckpt/model.h5'
    sentence = 'I can go out with my friend because i have to go bus'
    sentence = process_data(sentence)
    model = load_model(path_ckpt)
    print(model.predict(sentence))
