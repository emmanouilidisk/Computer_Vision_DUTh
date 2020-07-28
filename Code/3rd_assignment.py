import os
import cv2 as cv
import numpy as np
import json

image_db = "imagedb_train"
image_db_test = "imagedb_test"

sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

def find_bow_desc(desc, vocab):
    bow_desc = np.zeros((1, vocab.shape[0]), dtype = np.float32)
    for i in range(desc.shape[0]):
        diff = desc[i, :] - vocab
        dists = np.sum(np.square(diff), axis=1)
        min_dist_index = np.argmin(dists)
        # Increase frequency of the word in the histogram
        bow_desc[0, min_dist_index] += 1
    # Return histogram
    return bow_desc

def classify_with_svm(histogram, list_with_svms):
    data_expanded = np.expand_dims(histogram, axis=1)
    data_expanded_transpose = np.transpose(data_expanded)
    min_pred = 10000000000000000
    for elem in list_with_svms:
        prediction = elem.predict(data_expanded_transpose.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
        if prediction[0] <= min_pred:
            min_pred = prediction[0]
            best_svm = list_with_svms.index(elem)
    return best_svm


folders = os.listdir(image_db)
train_descs = np.zeros((0, 128))
print('Finding train_descs...')

for folder in folders:
    folder_path = os.path.join(image_db, folder)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        desc = extract_local_features(file_path)
        if desc is None:
            print("None")
            continue
        train_descs = np.concatenate((train_descs, desc), axis=0)

print('Creating vocabulary...')
term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
trainer = cv.BOWKMeansTrainer(50, term_crit, 1, cv.KMEANS_PP_CENTERS)
vocabulary = trainer.cluster(train_descs.astype(np.float32))
np.save('vocabulary.npy', vocabulary)

# Load vocabulary
vocabulary = np.load('vocabulary.npy')


#<------------------------------ Second Task ----------------------------------->
sift = cv.xfeatures2d_SIFT.create()
train_labels = np.zeros((0,1))
temp_label = np.zeros((1,1))

img_paths =[]

bow_descs = np.zeros((0, vocabulary.shape[0]))
for folder in folders:
    folder_path = os.path.join(image_db, folder)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        desc = extract_local_features(file_path)
        bow_desc = find_bow_desc(desc, vocabulary)
        img_paths.append(file_path)
        # computing histogram of image:
        bow_descs = np.concatenate((bow_descs, bow_desc), axis = 0)
        # labeling each image:
        train_labels = np.concatenate((train_labels, temp_label), axis = 0)
    temp_label[0] = temp_label[0] + 1
np.save('bow_descs.npy', bow_descs)
np.save('train_labels.npy', train_labels)


# Load bow_descs and train_labels
bow_descs = np.load('bow_descs.npy')
train_labels = np.load('train_labels.npy')
print('Found bow_descs & train_labels')

img_paths =[]


test_bow_descs = np.zeros((0, vocabulary.shape[0]))
test_labels = np.zeros((0,1))
temp_label = np.zeros((1,1))
folders = os.listdir(image_db_test)
for folder in folders:
    folder_path = os.path.join(image_db, folder)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        desc = extract_local_features(file_path)
        test_bow_desc = find_bow_desc(desc, vocabulary)
        img_paths.append(file_path)
        # computing histogram of test image:
        test_bow_descs = np.concatenate((test_bow_descs, test_bow_desc), axis = 0)
        # labeling each test image:
        test_labels = np.concatenate((test_labels, temp_label), axis = 0)
    temp_label[0] = temp_label[0] + 1
np.save('test_bow_descs.npy', test_bow_descs)
np.save('test_labels.npy', test_labels)


# Load test_bow_descs and test_labels
test_bow_descs = np.load('test_bow_descs.npy')
test_labels = np.load('test_labels.npy')
print('Found test_bow_descs & test_labels')

#<-------------------------------- Task 3 & 4 ------------------------------------->

#
# kNN algorithm
#
print("Starting kNN ...")

k = 2
counter = 0

for i in range(test_labels.shape[0]):
    sum_list = np.zeros(6, dtype=int)
    dists = np.sum((test_bow_descs[i] - bow_descs) ** 2, axis=1)
    sorted_ids = np.argsort(dists)
    for j in range (k):
        sum_list[int(train_labels[sorted_ids[j]])] += 1

    pred_label = np.argmax(sum_list, axis=0)
    #print('Prediction '+str(pred_label)+ 'Label '+str(test_labels[i]))
    if pred_label  == test_labels[i]:
        counter += 1
print("Success ratio "+str(100*counter/len(test_labels))+" %")

#
# SVM algorithm
#
print("Starting SVM ...")


# Create SVM classifiers
for i in range (6):
    temp_labels = np.zeros((train_labels.shape),dtype=int)
    for j in range (train_labels.shape[0]):
        if  i == train_labels[j]:
            temp_labels[j] = 1

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_RBF)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, temp_labels)
    svm.save("SVM" + str(i))
    print ("Created SVM" + str(i))

# Test SVM Model
counter = 0
SVMs_list = None

for i in range (6):
    loaded_svm = cv.ml.SVM_load("SVM" + str(i))
    if SVMs_list is None:
        SVMs_list = [loaded_svm]
    else:
        SVMs_list.append(loaded_svm)

for i in range (test_bow_descs.shape[0]):
    predicted_label = classify_with_svm(test_bow_descs[i], SVMs_list)
    #print('Predicted label = ', str(predicted_label), 'Correct label = ', str(test_labels[i]))
    if predicted_label == test_labels[i]:
        counter += 1
percentage = 100 *counter/test_labels.shape[0]

print("The success ratio is: "+ str(percentage) +" %")
