from load_data import process_data
from transformer.transformer_model import Transformers
from transformers import AutoTokenizer

def get_model(path):

    # load Tokenizer
    VietTokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    EngTokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print('Load token done!!! \n')

    vocab_input = EngTokenizer.vocab_size
    vocab_target = VietTokenizer.vocab_size

    # load model
    transformer = Transformers(source_token=EngTokenizer,
                                target_token=VietTokenizer,
                               vocab_input=vocab_input,
                               vocab_target=vocab_target,
                               training=False,
                               maxlen=128)
    transformer.build_model()
    transformer.load_model(path)

    return transformer

if __name__ == '__main__':
    path_ckpt = '/content/drive/MyDrive/translation/transformer.h5'
    sentence = [
        "When I was seven years old , I saw my first public execution , but I thought my life in North Korea was normal .",
        "I was so scared , I thought my heart was going to explode .",
        "Even though they were caught , they were eventually released after heavy international pressure .",
        "This was one of the lowest points in my life ."
    ]

    label = [
        "Khi tôi lên 7 , tôi chứng kiến cảnh người ta xử bắn công khai lần đầu tiên trong đời , nhưng tôi vẫn nghĩ cuộc sống của mình ở đây là hoàn toàn bình thường .",
        "Tôi đã vô cùng sợ hãi , và có cảm giác như tim mình sắp nổ tung .",
        "vì mặc dù đã bị bắt , nhưng cuối cùng học cũng được thả ra nhờ vào sức ép từ cộng đồng quốc tế .",
        "Đó là thời điểm tuyệt vọng nhất trong cuộc đời tôi .",
    ]

    GG = [
        "Khi tôi 7 tuổi, lần đầu tiên tôi bị hành quyết công khai, nhưng tôi nghĩ cuộc sống của mình ở Triều Tiên vẫn bình thường.",
        "Tôi đã rất sợ hãi, tôi nghĩ rằng trái tim mình sẽ nổ tung.",
        "Mặc dù họ đã bị bắt, cuối cùng họ đã được thả sau áp lực quốc tế nặng nề.",
        "Đây là một trong những điểm thấp nhất trong cuộc đời tôi."
    ]

    sentence = [process_data(text) for text in sentence]
    model = get_model(path_ckpt)
    print("\n\n---------------------------- PREDICT----------------------------")
    for i in range(len(GG)):
        print("Input: ", sentence[i])
        print("Label: ", label[i])
        print("GG translate: ", GG[i])
        print("Predict", model.predict(sentence[i]))
        print("\n----------------------------------------------------------------\n")
