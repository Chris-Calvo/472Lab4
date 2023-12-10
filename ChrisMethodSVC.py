from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
import os

from ChrisProcessing import load_data, extract_minutiae, calculate_minutiae_distances

def main():

    training_dir = 'C:/Users/Chris/Documents/School/4th Year/472Bio/TRAIN'
    testing_dir = 'C:/Users/Chris/Documents/School/4th Year/472Bio/TEST'
    # print('Loading Training Data...')
    training_data = load_data(training_dir)
    # print('Loading Testing Data...')
    testing_data = load_data(testing_dir)

    X_train, y_train = [], []
    X_test, y_test = [], []


    pbar = tqdm(desc='Training', total=1500)
    for pair in training_data:
      pbar.update(1)
      image1_path = os.path.join(training_dir, pair[0])
      image2_path = os.path.join(training_dir, pair[1])
      minutiae1 = extract_minutiae(image1_path)
      minutiae2 = extract_minutiae(image2_path)
      distances = calculate_minutiae_distances(minutiae1, minutiae2)
      X_train.append(distances)
      avg_dist = sum(distances) / len(distances)
      if avg_dist < 220:
          y_train.append(1)
      else:
          y_train.append(0)
    pbar.close()

    pbar2 = tqdm(desc="Testing...", total=500)
    for pair in testing_data:
        pbar2.update(1)
        image1_path = os.path.join(testing_dir, pair[0])
        image2_path = os.path.join(testing_dir, pair[1])
        minutiae1 = extract_minutiae(image1_path)
        minutiae2 = extract_minutiae(image2_path)
        distances = calculate_minutiae_distances(minutiae1, minutiae2)
        X_test.append(distances)
        avg_dist = sum(distances) / len(distances)
        if avg_dist < 220:
            y_test.append(1)
        else:
            y_test.append(0)
    pbar2.close()

    print('Modeling...')
    X_train = np.array(X_train).reshape(len(X_train), -1)
    X_test = np.array(X_test).reshape(len(X_test), -1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    ''' Comment in for average values, out for single run or hybrid '''

    # def run_model(X_train, y_train, X_test, y_test):
    #     model = SVC(kernel='linear')
    #     model.fit(X_train, y_train)
    #     predictions = model.predict(X_test)
    #     cm = confusion_matrix(y_test, predictions)
    #     tn, fp, fn, tp = cm.ravel()
    #     frr = fn / (fn + tp)
    #     far = fp / (fp + tn)
    #     return frr, far
    
    # frr_values = []
    # far_values = []

    # for _ in range(10):
    #     frr, far = run_model(X_train, y_train, X_test, y_test)
    #     frr_values.append(frr)
    #     far_values.append(far)

    # max_frr = max(frr_values)
    # min_frr = min(frr_values)
    # avg_frr = sum(frr_values) / len(frr_values)

    # max_far = max(far_values)
    # min_far = min(far_values)
    # avg_far = sum(far_values) / len(far_values)

    # print(f"Max FRR: {max_frr}, Min FRR: {min_frr}, Avg FRR: {avg_frr}")
    # print(f"Max FAR: {max_far}, Min FAR: {min_far}, Avg FAR: {avg_far}")


    ''' Comment out for average values or hybrid, in for single run'''

    # model = SVC(kernel='linear')
    # model.fit(X_train, y_train)

    # predictions = model.predict(X_test)
    
    # cm = confusion_matrix(y_test, predictions)
    # tn, fp, fn, tp = cm.ravel()
    # frr = fn/(fn+tp)
    # far = fp / (fp+tn)

    # eer = (frr+far)/2
    # print(f"False Reject Rate (FRR): {frr}")
    # print(f"False Accept Rate (FAR): {far}")
    # print(f"Equal Error Rate (EER): {eer}")
    # accuracy = accuracy_score(y_test, predictions)
    # print(f"Accuracy: {accuracy}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()