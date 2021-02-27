import transformers
from transformers import BertGenerationTokenizer, BertGenerationConfig, BertGenerationDecoder
import torch
import torch.nn as nn
import numpy as np

configuration = BertGenerationConfig(is_decoder=True)

model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder', is_decoder=True)
tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
random_size = (1, 512)


def getRandomTensor():
    multiplier = np.random.randint(low=1, high=28996)
    random_vec = multiplier*torch.rand(random_size)
    random_vec = random_vec.long()
    return random_vec


def readFile(filename):
    file = open(filename)
    file_contents = file.read()
    contents_split = file_contents.splitlines()
    return contents_split


v = readFile('bert-base-cased-vocab.txt')

if __name__ == '__main__':
    sent = []
    for i in range(10):
        print(i)
        random_vector = getRandomTensor()
        output = model(random_vector)
        print(output.shape)
        softmax = nn.Softmax(dim=1)
        prob = softmax(output.logits.squeeze())
        idx = torch.argmax(torch.argmax(prob, dim=0)).item()
        if idx > 28996:
            idx = np.random.randint(low=1, high=30522)
        sent.append(v[idx])
    print(" ".join(sent))
