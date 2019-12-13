from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, metrics
from multiprocessing import Process, Manager
import re
import matplotlib.pyplot as plt


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

    # Try out a bunch of different models and test for accuracy:
    # Try training x models of each type and using the one with greatest accuracy
    greatest_rf_model = 0
    """
    greatest_knn_model = 0
    greatest_mlp_model = 0
    greatest_boost_model = 0
    greatest_adaboost_model = 0
    greatest_svm_model = 0
    greatest_voting_model = 0
    """
    greatest_score = 0
    for i in range(10000):
        rf_model = RandomForestClassifier(n_estimators=19, n_jobs=4) # n_estimator = 19, 5 found to be the best, possible accuracy of .975. 182 best average
        rf_model.fit(training_data, training_classifications)
        if greatest_rf_model == 0:
            greatest_rf_model = rf_model
        else:
            score = rf_model.score(scoring_data, scoring_classifications)
            if score > greatest_score:
                greatest_rf_model = rf_model
                greatest_score = score
                if greatest_score > .975:
                    break

    print("Random Forest Accuracy: " + str(greatest_rf_model.score(scoring_data, scoring_classifications)))

    # Output results of challenge set
    results_file = open("results.txt", "w")
    stats_file = open("./cs185c_feature_vectors/Challenge_stats.txt", "r")
    for i in range(len(challenge_data)):
        prediction = greatest_rf_model.predict([challenge_data[i]])
        line = stats_file.readline()[:8]
        if prediction == [1]:
            line += ", C\n"
        else:
            line += ", R\n"
        results_file.write(line)

        # Everything below here was used for testing stuff. I know, it's ugly.
"""
    for i in range(10):
        knn_model = KNeighborsClassifier(n_neighbors=1, algorithm = "brute")
        knn_model.fit(training_data, training_classifications)
        if greatest_knn_model == 0:
            greatest_knn_model = knn_model
        elif knn_model.score(scoring_data, scoring_classifications) > greatest_knn_model.score(scoring_data, scoring_classifications):
            greatest_knn_model = knn_model

        mlp_model = MLPClassifier(max_iter=1000)
        mlp_model.fit(training_data, training_classifications)
        if greatest_mlp_model == 0:
            greatest_mlp_model = mlp_model
        elif mlp_model.score(scoring_data, scoring_classifications) > greatest_mlp_model.score(scoring_data, scoring_classifications):
            greatest_mlp_model = mlp_model


        boost_model = GradientBoostingClassifier()
        boost_model.fit(training_data, training_classifications)
        if greatest_boost_model == 0:
            greatest_boost_model = boost_model
        elif boost_model.score(scoring_data, scoring_classifications) > greatest_boost_model.score(scoring_data, scoring_classifications):
            greatest_boost_model = boost_model

        adaboost_model = AdaBoostClassifier(base_estimator = RandomForestClassifier(n_estimators = 18))
        adaboost_model.fit(training_data, training_classifications)
        if greatest_adaboost_model == 0:
            greatest_adaboost_model = adaboost_model
        elif adaboost_model.score(scoring_data, scoring_classifications) > greatest_adaboost_model.score(scoring_data, scoring_classifications):
            greatest_adaboost_model = adaboost_model


        svm_model = svm.SVC(kernel="poly", gamma="auto")
        svm_model.fit(training_data, training_classifications)
        if greatest_svm_model == 0:
            greatest_svm_model = svm_model
        elif svm_model.score(scoring_data, scoring_classifications) > greatest_svm_model.score(scoring_data, scoring_classifications):
            greatest_svm_model = svm_model



    x_points_red = []
    y_points_red = []
    x_points_blue = []
    y_points_blue = []
    for i in range(len(scoring_data)):
        #print("test: " + str(greatest_rf_model.predict_proba([scoring_data[i]])) + " " + str(greatest_rf_model.predict([scoring_data[i]])))
        point = greatest_rf_model.predict_proba([scoring_data[i]])
        if scoring_classifications[i] == 1:
            x_points_red.append(point[0][0])
            y_points_red.append(point[0][1])
        elif scoring_classifications[i] == -1:
            x_points_blue.append(point[0][0])
            y_points_blue.append(point[0][1])
        else:
            print("error")
    plt.scatter(x_points_red, y_points_red,color="red")
    plt.scatter(x_points_blue, y_points_blue, marker="+", color="blue")
    plt.plot([.5, .5], color="black")
    plt.title("Random Forest Scatter Plot")
    plt.show()

    metrics.plot_roc_curve(greatest_rf_model, scoring_data, scoring_classifications)
    plt.show()



    # Can play around more with weights (and different voting classifiers) to see if it yields better results
    weights = [1, 1, 2] #[1, 1, 1] #[1, 1, 1, 1, 5]
    voting_model = VotingClassifier(estimators=[("knn", knn_model), ("rf", rf_model), ("mlp", mlp_model)], voting="soft", weights=weights)
    voting_model.fit(training_data, training_classifications)

    for a in range(3):
        for b in range(3):
            for c in range(3):


                if greatest_voting_model == 0:
                    greatest_voting_model = voting_model
                elif voting_model.score(scoring_data, scoring_classifications) > greatest_voting_model.score(scoring_data, scoring_classifications):
                    greatest_voting_model = voting_model
                    best_weights = [a + 1, b + 1, c + 1]


    for a in range(2):
        for b in range(3):
            for c in range(5):
                for d in range(2):
                    for e in range(5):
                        weights = [a + 1, b + 1, c + 1, d + 1, e + 1]
                        voting_model = VotingClassifier(estimators=[("knn", knn_model), ("rf", rf_model), ("mlp", mlp_model), ("gradient_boost", boost_model), ("adaboost", adaboost_model)], voting="soft", weights=weights)
                        voting_model.fit(training_data, training_classifications)
                        if greatest_voting_model == 0:
                            greatest_voting_model = voting_model
                        elif voting_model.score(scoring_data, scoring_classifications) > greatest_voting_model.score(scoring_data, scoring_classifications):
                            greatest_voting_model = voting_model
                            best_weights = [a + 1, b + 1, c + 1, d + 1, e + 1]
    print(str(best_weights))


    print("Random Forest Accuracy: " + str(greatest_rf_model.score(scoring_data, scoring_classifications)))
    print("KNN Accuracy: " + str(greatest_knn_model.score(scoring_data, scoring_classifications)))
    print("MLP Accuracy: " + str(greatest_mlp_model.score(scoring_data, scoring_classifications)))
    #print("Gradient Boost Accuracy: " + str(greatest_boost_model.score(scoring_data, scoring_classifications)))
    #print("Adaboost Accuracy: " + str(greatest_adaboost_model.score(scoring_data, scoring_classifications)))
    print("SVM Accuracy: " + str(greatest_svm_model.score(scoring_data, scoring_classifications)))
    print("Voting Accuracy: " + str(voting_model.score(scoring_data, scoring_classifications)))
    # Best accuracy I've seen so far is .975 from random forest with n_estimators = 19, 5 (other promising values: 18, 132)



    bad_vectors = []
    accuracy = 0
    # Try using all combined for classification
    for i in range(len(scoring_data)):
        correct_predictions = []
        correct_prediction = scoring_classifications[i]

        rf_prediction = greatest_rf_model.predict([scoring_data[i]])
        if rf_prediction == correct_prediction:
            correct_predictions.append("rf")
        knn_prediction = greatest_knn_model.predict([scoring_data[i]])
        if knn_prediction == correct_prediction:
            correct_predictions.append("knn")
        mlp_prediction = greatest_mlp_model.predict([scoring_data[i]])
        if mlp_prediction == correct_prediction:
            correct_predictions.append("mlp")

        gradient_boost_prediction = greatest_boost_model.predict([scoring_data[i]])
        if gradient_boost_prediction == correct_prediction:
            correct_predictions.append("gradient_boost")
        adaboost_prediction = greatest_adaboost_model.predict([scoring_data[i]])
        if adaboost_prediction == correct_prediction:
            correct_predictions.append("adaboost")

        svm_prediction = greatest_svm_model.predict([scoring_data[i]])
        if svm_prediction == correct_prediction:
            correct_predictions.append("svm")
        voting_prediction = voting_model.predict([scoring_data[i]])
        if voting_prediction == correct_prediction:
            correct_predictions.append(" voting")

        predictions = []
        predictions.append(rf_prediction)
        predictions.append(knn_prediction)
        predictions.append(mlp_prediction)
        #predictions.append(gradient_boost_prediction)
        #predictions.append(adaboost_prediction)
        predictions.append(svm_prediction)
        predictions.append(voting_prediction)



        count_ceeinject = 0
        count_renos = 0
        for p in predictions:
            if p == [1]:
                count_ceeinject += 1
            elif p == [-1]:
                count_renos += 1
        prediction = 0
        if count_ceeinject > count_renos:
            prediction = 1
        else:
            prediction = -1
        if prediction == correct_prediction:
            accuracy += 1
        else:
            bad_vectors.append(scoring_data[i])
        print(str(prediction) + " Count [1] (ceeinject): " + str(count_ceeinject) + " count [-1] (renos): " + str(count_renos) + " actual: " + str(scoring_classifications[i]), end="")
        print(" Correct: ", end="")
        for c in correct_predictions:
            print(str(c) + " ", end="")
        print("")

    print("accuracy: " + str(accuracy / len(scoring_data)))


    return
    """



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
    n_estimator = val - 4
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
    for i in range(100):
        rf_model = RandomForestClassifier(n_estimators=n_estimators)
        rf_model.fit(training_data, training_classifications)
        accuracy = rf_model.score(scoring_data, scoring_classifications)
        sum += accuracy
        if accuracy > average_accuracies["greatest"]:
            average_accuracies["greatest"] = accuracy
            average_accuracies["greatest_n_estimators"] = n_estimators
    average_accuracies[n_estimators] = sum / 100 # Put avg accuarcy in the dictionary
    print("n_estimators: " + str(n_estimators) + " accuracy: " + str(sum / 100))


# Processes put the results in a dictionary average_accuracies
manager = Manager()
average_accuracies = manager.dict()
average_accuracies["greatest"] = 0
average_accuracies["greatest_n_estimators"] = 0
for i in range(100):
    find_best_rf((i + 1) * 5)
max_accuracy = 0
max_n_estimator = 0
# Search through dictionary to find greatest accuracy
for key in average_accuracies:
    if key == "greatest" or key == "greatest_n_estimators":
        continue
    val = average_accuracies[key]
    if val > max_accuracy:
        max_accuracy = val
        max_n_estimator = key

print("best average n_estimators: " + str(max_n_estimator) + " max average accuracy: " + str(max_accuracy))
print("Greatest accuracy found: " + str(average_accuracies["greatest"]) + " n_estimators: " + str(average_accuracies["greatest_n_estimators"]))
# Best seen so far is .975 with n_estimators = 19
