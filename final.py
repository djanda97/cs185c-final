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


def get_similarities(word_vectors, opcode_pairs):
    similarities = {}

    for pair in opcode_pairs:
        similarities[pair] = word_vectors.similarity(pair[0], pair[1])

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
        self.epoch = 1

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
    # data = PathLineSentences(path)

    epoch_logger = EpochLogger()

    # if not os.path.exists(args.family + ".model"):
    model = Word2Vec(data, size=2, window=6, min_count=1,
                        workers=4, callbacks=[epoch_logger])
    model_file = args.family + ".model"
    model.save(model_file)
    # else:
    #     model = Word2Vec.load(args.family + ".model")

    # model = Word2Vec.load(model_file)

    # samples = []
    # models = []

    # for i, file in enumerate(os.listdir(path)):
    #     samples.append(LineSentence(file))
    #     models.append(Word2Vec(samples[i], size=2, window=6, min_count=1,
    #                            workers=4, callbacks=[epoch_logger]))

    # for s in samples:
    #     print(s)

    # for m in models:
    #     print(m)

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


if __name__ == "__main__":
    run()
