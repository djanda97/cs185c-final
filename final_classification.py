from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Process, Manager
import re


# Function to get training, scoring, and challenge data. Returns them each in the form of a list of vectors.
# Also returns classifications for training and scoring.
def get_data():
    training_data_size = 1600
    scoring_data_size = 200
    ceeinject_sample_count = 900
    renos_sample_count = 900
    challenge_sample_count = 200
    vector_length = 40 # Expected length of each vector
    input_file_name = "./result.txt"

    # Initialize our data sets (each contains a list of lists/vectors)
    training_data = [[] for j in range(training_data_size)]
    scoring_data = [[] for j in range(scoring_data_size)]
    challenge_data = [[] for j in range(challenge_sample_count)]

    # Open our input files.
    ceeinject_file = open("./cs185c_feature_vectors/CeeInject_w2v.txt", "r")
    renos_file = open("./cs185c_feature_vectors/Renos_w2v.txt", "r")
    challenge_file = open("./cs185c_feature_vectors/Challenge_w2v.txt", "r")

    training_index = 0
    scoring_index = 0
    # First fill in data for ceeinject. Should be 450 samples for training, 450 for scoring
    for i in range(ceeinject_sample_count):
        line = ceeinject_file.readline()
        line = re.sub("\[|\]|,|\n", "", line)
        vector = line.split(" ")[:40]
        vector = [float(i) for i in vector] # Convert strings to floats
        if len(vector) != 40:
            print("Error in data, vector length is " + str(len(vector)))
        if i < training_data_size / 2:
            training_data[training_index] = vector
            training_index += 1
        else:
            scoring_data[scoring_index] = vector
            scoring_index += 1
    # Now repeat process for renos.
    for i in range(renos_sample_count):
        line = renos_file.readline()
        line = re.sub("\[|\]|,|\n", "", line)
        vector = line.split(" ")[:40]
        vector = [float(i) for i in vector] # Convert strings to floats
        if len(vector) != 40:
            print("Error in data, vector length is " + str(len(vector)))
        if i < training_data_size / 2:
            training_data[training_index] = vector
            training_index += 1
        else:
            scoring_data[scoring_index] = vector
            scoring_index += 1
    # Lastly fill in challenge data
    for i in range(challenge_sample_count):
        line = challenge_file.readline()
        line = re.sub("\[|\]|,|\n", "", line)
        vector = line.split(" ")[:40]
        vector = [float(i) for i in vector] # Convert strings to floats
        challenge_data[i] = vector
        if len(vector) != 40:
            print("Error in data, vector length is " + str(len(vector)))

    # Now fill a list of classifications, 1 for ceeinject, -1 for renos
    training_classifications = [1 for i in range(800)] + [-1 for i in range(800)]
    scoring_classifications = [1 for i in range(100)] + [-1 for i in range(100)]

    return [training_data, training_classifications, scoring_data, scoring_classifications, challenge_data]



def run():

    [training_data, training_classifications, scoring_data, scoring_classifications, challenge_data] = get_data()

    # Build our random forest model and test the accuracy
    # n_estimator = 132 found to be the best
    rf_model = RandomForestClassifier(n_estimators=132)
    rf_model.fit(training_data, training_classifications)
    print(rf_model.score(scoring_data, scoring_classifications))


if __name__ == "__main__":
    run()








exit()
# The code below here is used to find the optimum random forest, i.e. that with the greatest accuracy (by varying n_estimators).
# It builds random forests in separate processes to try to speed things up by running them in parallel.

# Function to find the optimum random forest. Might take a while. Batch processes to run 10 at a time.
def find_best_rf(val):
    [training_data, training_classifications, scoring_data, scoring_classifications, challenge_data] = get_data()

    # Make our program run multiple instances in parallel
    processes = []
    n_estimator = val - 9
    while n_estimator <= val:
        process = Process(target = find_average_accuracy_rf, args = (n_estimator, training_data, training_classifications, scoring_data, scoring_classifications))
        process.start()
        processes.append(process)
        n_estimator += 1

    # Wait for processes to finish
    for process in processes:
        process.join()

# Function that builds 10 random forests with the same parameters and find it's average accuracy
def find_average_accuracy_rf(n_estimators, training_data, training_classifications, scoring_data, scoring_classifications):
    sum = 0
    for i in range(10):
        rf_model = RandomForestClassifier(n_estimators=n_estimators)
        rf_model.fit(training_data, training_classifications)
        sum += rf_model.score(scoring_data, scoring_classifications)
    average_accuracies[n_estimators] = sum / 10 # Put avg accuarcy in the dictionary
    print("n_estimators: " + str(n_estimators) + " accuracy: " + str(sum / 10))


# Processes put the results in a dictionary average_accuracies
manager = Manager()
average_accuracies = manager.dict()
for i in range(100):
    find_best_rf((i + 1) * 10)
max_accuracy = 0
max_n_estimator = 0
# Search through dictionary to find greatest accuracy
for key, val in average_accuracies:
    if val > max_accuracy:
        max_accuracy = val
        max_n_estimator = key

print("best n_estimators: " + str(max_n_estimator) + " max accuracy: " + str(max_accuracy))
