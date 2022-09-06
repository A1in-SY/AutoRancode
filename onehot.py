import torch

rancode_array = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
rancode_size = 4


def text2vec(text):
    vec = torch.zeros(rancode_size, rancode_array.__len__())
    for i in range(len(text)):
        vec[i][rancode_array.index(text[i])] = 1
    return vec


def vec2text(vec):

    vec = torch.argmax(vec, dim=1)
    text = ""
    for v in vec:
        text += rancode_array[v]
    return text


if __name__ == '__main__':
    vec = text2vec("test")
    print(vec.shape)
    print(vec2text(vec))
