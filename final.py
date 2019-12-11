from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from itertools import combinations
import argparse
import pickle
import os.path


def get_similarities(word_vectors, opcode_pairs):
    similarities = []

    for pair in opcode_pairs:
        similarity = word_vectors.similarity(pair[0], pair[1])
        similarities.append(similarity)

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

        top20 = [opcode.lower() for opcode in top20]

        opcode_pairs = combinations(top20, 2)

        with open("opcode_pairs.pickle", "wb") as file:
            pickle.dump(opcode_pairs, file)
    else:
        with open("opcode_pairs.pickle", "rb") as file:
            opcode_pairs = pickle.load(file)

    return opcode_pairs


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def run():
    parser = argparse.ArgumentParser(
        description="Compute embedding vectors for different malware families.")
    parser.add_argument(
        "--family", "-f", help="Specify which family of malware to compute embedding vectors for. (CeeInject, Renos, or Challenge)", type=str, default="CeeInject")

    args = parser.parse_args()

    path = "./cs185c_final_data/" + args.family

    data = PathLineSentences(path, limit=100)

    epoch_logger = EpochLogger()

    model = Word2Vec(data, size=2, window=6, min_count=1,
                     workers=4, callbacks=[epoch_logger])
    model_file = args.family + ".model"
    model.save(model_file)

    # model = Word2Vec.load(model_file)

    # print("training:", model.train(
    #     [["mov", "sub", "pop", "push"]], total_examples=10, epochs=3))

    word_vectors = model.wv
    # vec_mov = model.wv["mov"]
    # print("word_vectors:", word_vectors.get_vector("mov"))
    # print("vec_mov:", vec_mov)

    opcode_pairs = get_opcode_pairs()

    similarities = get_similarities(word_vectors, opcode_pairs)

    print("similarities:\n", similarities)

    # for i, pair in enumerate(opcode_pairs):
    #     # print(pair[0], pair[1])
    #     print(f"{i + 1}:", pair)


if __name__ == "__main__":
    run()
