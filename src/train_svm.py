import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle


def get_X_Y(data, verb=False, split=""):
    data = data[1:, :]

    X = data[:, :-1]
    Y = data[:, -1]
    unique_values, counts = np.unique(Y, return_counts=True)

    if verb: 
        classes = {0: "not trash", 1: "trash"}
        print(f"Examples in {split} set:")
        for value, count in zip(unique_values, counts):
            print(f"{classes[value]}: {count}")
        
    return X, Y


def print_scores(X, Y, classifier, split=""):
    Y_pred = classifier.predict(X)
    accuracy = accuracy_score(Y, Y_pred)
    precision = precision_score(Y, Y_pred)
    recall = recall_score(Y, Y_pred)
    f1 = f1_score(Y, Y_pred)

    print(f"For {split} set:")
    print(f"accuracy_score: {accuracy:.2f}")
    print(f"precision: {precision:.2f}")
    print(f"recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print("-------------------")    


    return f"{accuracy:.2f},{precision:.2f},{recall:.2f},{f1:.2f},"

def train_and_test(verb=False):
    train_data = np.loadtxt('data/feature_vectors_training.csv', delimiter=',')
    val_data = np.loadtxt('data/feature_vectors_validation.csv', delimiter=',')
    test_data = np.loadtxt('data/feature_vectors_test.csv', delimiter=',')

    X_train, Y_train = get_X_Y(train_data, verb, split="train")
    X_val, Y_val = get_X_Y(val_data, verb, split="validation")
    X_test, Y_test = get_X_Y(test_data, verb, split="test")

    print("\n")

    svm_classifier = SVC(kernel='rbf', random_state=42, class_weight="balanced", C=1.2)

    # Without scaler
    print("WITHOUT SCALER\n")
    svm_classifier.fit(X_train, Y_train)

    scores_string = ""
    scores_string += print_scores(X_train, Y_train, svm_classifier, split="train")
    scores_string += print_scores(X_val, Y_val, svm_classifier, split="validation")
    scores_string += print_scores(X_test, Y_test, svm_classifier, split="test")
    print(scores_string)


    # With scaler
    print("\n\nWITH SCALER\n")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    svm_classifier.fit(X_train_scaled, Y_train)

    scores_string = ""
    scores_string += print_scores(X_train_scaled, Y_train, svm_classifier, split="train")
    scores_string += print_scores(X_val_scaled, Y_val, svm_classifier, split="validation")
    scores_string += print_scores(X_test_scaled, Y_test, svm_classifier, split="test")
    print(scores_string)

if __name__ == "__main__":
    # train_and_test(verb=True)

    # train and save the best model
    train_data = np.loadtxt('data/feature_vectors_training.csv', delimiter=',')
    X_train, Y_train = get_X_Y(train_data, verb=False, split="train")

    svm_classifier = SVC(kernel='rbf', random_state=42, class_weight="balanced", C=1.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svm_classifier.fit(X_train_scaled, Y_train)
    
    with open('out/svc_model.pkl', 'wb') as f:
        pickle.dump((scaler, svm_classifier), f)
        # pickle.dump(svm_classifier, f)
    
    print("Model saved to out/svc_model.pkl")


