import cv2
#from skimage import metrics
#import numpy as np
import os

predictions = []
TRAIN_path = 'C:/Users/Chris/Documents/School/4th Year/472Bio/J_TRAIN'
TRAINCOMPARE_path = 'C:/Users/Chris/Documents/School/4th Year/472Bio/J_TRAINCOMPARE'

def compare_same(filename, degree_of_error):
    image1_pre = cv2.imread(TRAIN_path + '/' + filename)
    ret,thresh = cv2.threshold(image1_pre,90,255,cv2.THRESH_BINARY)
    image1 = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #image1 = cv2.normalize(image1_pre, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow('Normalized Image 1', image1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image2_pre = cv2.imread(TRAINCOMPARE_path + '/s' + filename[1:])
    ret,thresh = cv2.threshold(image2_pre,90,255,cv2.THRESH_BINARY)
    image2 = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #image2 = cv2.normalize(image2_pre, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow('Normalized Image 2', image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    try:
        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},{}).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []

        for x, y in matches:
            if (x.distance + (x.distance * degree_of_error) > y.distance) and (x.distance - (x.distance * degree_of_error) < y.distance):
                match_points.append(x)

        keypoints = 0
        if len(keypoints_1) < len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        # result = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, match_points, None)
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        score = len(match_points) / keypoints * 100
        if score >= 75:
            predictions.append(1)
            return True
        else:
            predictions.append(0)
            return False
    except:
        predictions.append(0)
        return False
    
def compare_diff(filename_a, filename_b, degree_of_error):
    image1_pre = cv2.imread(TRAIN_path + '/' + filename_a)
    ret,thresh = cv2.threshold(image1_pre,90,255,cv2.THRESH_BINARY)
    image1 = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #image1 = cv2.normalize(image1_pre, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow('Normalized Image 1', image1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image2_pre = cv2.imread(TRAINCOMPARE_path + '/' + filename_b)
    ret,thresh = cv2.threshold(image2_pre,90,255,cv2.THRESH_BINARY)
    image2 = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #image2 = cv2.normalize(image2_pre, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow('Normalized Image 2', image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    try:
        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},{}).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []

        for x, y in matches:
            if (x.distance + (x.distance * degree_of_error) > y.distance) and (x.distance - (x.distance * degree_of_error) < y.distance):
                match_points.append(x)

        keypoints = 0
        if len(keypoints_1) < len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        # result = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, match_points, None)
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        score = len(match_points) / keypoints * 100
        if score >= 75:
            predictions.append(1)
            return True
        else:
            predictions.append(0)
            return False
    except:
        predictions.append(0)
        return False

def FAR_explorer(filename, degree_of_error):
    FAR_counter = 0
    mod_counter = 0
    for s_filename in os.listdir(TRAINCOMPARE_path):
        if mod_counter % 500 == 0:
            f = os.path.join(TRAINCOMPARE_path, s_filename)
            split_tup = os.path.splitext(s_filename)
            # checking if valid file AND is png
            if os.path.isfile(f) and split_tup[1] == '.png':
                if filename[1:] != s_filename[1:]:
                    if compare_diff(filename, s_filename, degree_of_error):
                        FAR_counter += 1
            mod_counter += 1
    FAR_result = FAR_counter / 3
    return FAR_result

def FRR_FAR_Calculator(degree_of_error):
    FRR_counter = 0
    FAR_total = 0
    for filename in os.listdir(TRAIN_path):
        f = os.path.join(TRAIN_path, filename)
        split_tup = os.path.splitext(filename)
        # checking if valid file AND is png
        if os.path.isfile(f) and split_tup[1] == '.png':
            if not compare_same(filename, degree_of_error):
                FRR_counter += 1
            FAR_total += FAR_explorer(filename, degree_of_error)
    FRR_final = FRR_counter / 1500
    FAR_final = FAR_total / 1500

    return FRR_final, FAR_final

def main():

    doe = 0.01
    counter = 0
    FRR_avg = 0
    FRR_min = 2
    FRR_max = 0
    FAR_avg = 0
    FAR_min = 2
    FAR_max = 0
    EER_val = 2
    EER_doe = 0
    EER_fin = 0

    while doe < 0.26:
        FRR_val, FAR_val = FRR_FAR_Calculator(doe)

        # individual results
        # print("[" + str(doe) + "]  FRR = " + str(FRR_val) + ", FAR =" + str(FAR_val))

        #average calculations
        FRR_avg += FRR_val
        FAR_avg += FAR_val

        #minimum calculations
        if FRR_min > FRR_val:
            FRR_min = FRR_val
        if FAR_min > FAR_val:
            FAR_min = FAR_val

        #maximum calculations
        if FRR_max < FRR_val:
            FRR_max = FRR_val
        if FAR_max < FAR_val:
            FAR_max = FAR_val

        # Equal Error Rate
        EER_check = FRR_val - FAR_val
        if EER_check < 0:
            EER_check = EER_check * -1
        if EER_check < EER_val:
            EER_doe = doe
            EER_fin = (FRR_val + FAR_val) /2

        counter += 1
        doe += 0.01

    FRR_avg = FRR_avg / counter
    FAR_avg = FAR_avg / counter

    print("FRR Avg = " + str(FRR_avg))
    print("FRR Min = " + str(FRR_min))
    print("FRR Max = " + str(FRR_max))
    print("FAR Avg = " + str(FAR_avg))
    print("FAR Min = " + str(FAR_min))
    print("FAR Max = " + str(FAR_max))
    print("EER Avg = " + str(EER_fin) + " @ " + str(EER_doe))

    return predictions

if __name__  == "__main__":
    main()
