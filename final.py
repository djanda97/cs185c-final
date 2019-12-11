from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from itertools import combinations
import argparse
import pickle
import os.path


def print_opcodes(data, label):
    print(label)
    for opcode in data:
        print(opcode)


def get_similarities(word_vectors, opcode_pairs, label):
    similarities = []
    # for wv in word_vectors:
    #     similarities.append(word_vectors.similarity("mov", "sub"))
    return similarities


def get_top20(filename):
    file = open(filename, "r")
    top20 = file.read()

    removeChars = "[',]"

    for char in removeChars:
        top20 = top20.replace(char, "")

    top20 = top20.split()
    return top20


def get_opcode_pairs():
    if not os.path.exists("opcode_pairs.pickle"):
        top20 = get_top20("./cs185c_final_data/top20.txt")
        opcode_pairs = combinations(top20, 2)

        with open("opcode_pairs.pickle", "wb") as file:
            pickle.dump(opcode_pairs, file)
    else:
        with open("opcode_pairs.pickle", "rb") as file:
            opcode_pairs = pickle.load(file)

    return opcode_pairs


def run():
    parser = argparse.ArgumentParser(
        description="Compute embedding vectors for different malware families.")
    parser.add_argument(
        "--family", "-f", help="Specify which family of malware to compute embedding vectors for. (CeeInject, Renos, or Challenge)", type=str, default="CeeInject")

    args = parser.parse_args()

    path = "./cs185c_final_data/" + args.family

    data = PathLineSentences(path, limit=100)

    model = Word2Vec(data, size=2, window=6, min_count=1, workers=4)
    model_file = args.family + ".model"
    model.save(model_file)

    model = Word2Vec.load(model_file)

    print("training:", model.train(
        [["mov", "sub", "pop", "push"]], total_examples=10, epochs=3))

    word_vectors = model.wv
    print("word_vectors:", word_vectors.get_vector("mov"))

    opcode_pairs = get_opcode_pairs()

    for i, pair in enumerate(opcode_pairs):
        # print(pair[0], pair[1])
        print(f"{i + 1}:", pair)


if __name__ == "__main__":
    run()
