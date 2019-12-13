from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from itertools import combinations
from pprint import pprint
import argparse
import pickle
import os.path
import numpy as np


def get_similarities(word_vectors, opcode_pairs):
    similarities = {}
    positive_count = 0
    negative_count = 0

    for pair in opcode_pairs:
        similarity = word_vectors.similarity(pair[0], pair[1])
        similarities[pair] = similarity
        if similarity >= 0:
            positive_count += 1
        else:
            negative_count += 1

    print("\npositive_count:", positive_count)
    print("negative_count:", negative_count)
    print("difference:", abs(positive_count - negative_count))

    return similarities


def get_top20(filename):
    file = open(filename, "r")
    top20 = file.read()

    removeChars = "[',]"

    for char in removeChars:
        top20 = top20.replace(char, "")

    top20 = [opcode.lower() for opcode in top20.split()]

    return top20


def get_opcode_pairs():
    top20 = get_top20("./cs185c_final_data/top20.txt")
    opcode_pairs = combinations(top20, 2)

    return opcode_pairs


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
        self.epoch += 1


def create_model(family):
    path = "./cs185c_final_data/" + family
    data = PathLineSentences(path)
    epoch_logger = EpochLogger()

    model_file = family + ".model"

    if not os.path.exists(model_file):
        model = Word2Vec(data, size=2, window=6, min_count=1,
                         workers=4, callbacks=[epoch_logger])
        model.save(model_file)
    else:
        model = Word2Vec.load(model_file)

    word_vectors = model.wv

    opcode_vectors = []

    print("\nopcode_vectors:")
    for i, opcode in enumerate(word_vectors.vocab):
        opcode_vectors.append(word_vectors[opcode])
        print(f"{i + 1}:", opcode, opcode_vectors[i])

    opcode_pairs = get_opcode_pairs()

    similarities = get_similarities(word_vectors, opcode_pairs)

    print("\nsimilarities:")
    for pair, similarity in similarities.items():
        print(pair, ":", similarity)


def get_opcode_groups():
    print("Getting opcode groups:")


def problem_1a():
    print("\nProblem 1a:")
    create_model("CeeInject")


def problem_1b():
    print("\nProblem 1b:")
    create_model("Renos")


def create_model_samples(path):
    samples = []
    models = []

    for i, file in enumerate(sorted(os.listdir(path))):
        if i % 100 == 0:
            print(f"file #{i}")
        samples.append(LineSentence(path + file, limit=100))
        models.append(
            Word2Vec(samples[i], size=2, window=6, min_count=1, workers=4))

    return models


def get_embedding_vectors(models):
    embedding_vectors = np.array([])

    top20 = get_top20("./cs185c_final_data/top20.txt")

    for m in models:
        for i in range(len(top20)):
            opcode = top20[i]
            if opcode in m.wv.vocab:
                embedding_vectors = np.append(embedding_vectors, m.wv[opcode])
            else:
                embedding_vectors = np.append(embedding_vectors, [0, 0])
    
    embedding_vectors = np.reshape(embedding_vectors, (len(models), 40))

    return embedding_vectors


def print_embedding_vectors(embedding_vectors):
    for i in range(embedding_vectors.shape[0]):
        print(f"embedding_vectors[{i}]:")
        pprint(embedding_vectors[i])
        print(f"embedding_vectors[{i}].shape:", embedding_vectors[i].shape)

    for i in range(embedding_vectors.shape[0]):
        print(f"\nembedding_vectors[0][{i}]:")
        pprint(embedding_vectors[0][i])
        print(f"embedding_vectors[0][{i}].shape:",
            embedding_vectors[0][i].shape)


def problem_1c():
    print("\nProblem 1c:")

    paths = [
        "./cs185c_final_data/CeeInject/",
        "./cs185c_final_data/Renos/",
        "./cs185c_final_data/Challenge/"
    ]

    models = [
        create_model_samples(paths[0]),
        create_model_samples(paths[1]),
        create_model_samples(paths[2])
    ]

    # filename = "embedding_vectors.txt"
    # if not os.path.exists(filename):
    #     file = open(filename, "w")
    #     file.write()
    
    embedding_vectors = np.array([
        get_embedding_vectors(models[0]),
        get_embedding_vectors(models[1]),
        get_embedding_vectors(models[2])
    ])

    print_embedding_vectors(embedding_vectors)


if __name__ == "__main__":
    problem_1a()
    problem_1b()
    problem_1c()
