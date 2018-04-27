import Augmentor
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import load_model

def load_directory(directory):
    """Loads image files from the directory in
    advance of training and evaluation.
    Uses relative path (relative to current directory.)
    """
    os.system('rm -r train')
    os.system('rm -r eval')
    os.system('mkdir train')
    os.system('mkdir eval')
    chars = [directory+x for x in os.popen('ls {}'.format(directory)).read().split('\n')][:-1]
    columns = ['file', 'label']
    train = pd.DataFrame(columns=columns)
    train_path = './train/'
    test = pd.DataFrame(columns=columns)
    val_path = './eval/'
    for j, char in enumerate(chars):
        images = [x for x in os.popen('ls {}'.format(char)).read().split('\n')][:-1]
        for i, img in enumerate(images):
            #reserve first example for training
            if i == 0:
                train_img = Image.open(char + '/' + img)
                train_img.save(train_path + img)
                train_row = pd.DataFrame(columns=columns, index=[len(train)])
                train_row['file'] = img
                train_row['label'] = j
                train = pd.concat([train, train_row])
            #add remaining examples to eval directory
            elif i > 0:
                test_img = Image.open(char + '/' + img)
                test_img.save(val_path + img)
                test_row = pd.DataFrame(columns=columns, index=[len(test)])
                test_row['file'] = img
                test_row['label'] = j
                test = pd.concat([test, test_row])
    labels = train.label.unique()
    labels.sort()
    return train, test, labels


def as_tensor(image):
    """Converts an image object into a tensor
    shape (m, n, 1) (channels last)
    """
    as_array = np.array(image)
    result = np.expand_dims(as_array, axis=-1)
    return result

def image_from_index(idx, df, path):
    """Gets an image given an index. Requires dataframe
    and path to locate image.
    """
    filename = df.loc[idx]['file']
    img = Image.open(path + filename)
    return img

def augment(img, loss=1):
    """applies a series of augmentations to 
    input image and returns.
    """
    #use loss to control augmentation likelihood.
    if random.random() < loss:
        return img
    #Apply random skew to img
    skew = Augmentor.Operations.Skew(1, 'RANDOM', .5)
    res = skew.perform_operation([img])
    img = res[0]
    #Apply random elastic distortion to img
    distortion = random.randint(5, 16)
    gridx = random.randint(1, 3)
    gridy = random.randint(1, 3)
    distort = Augmentor.Operations.Distort(1, gridx, gridy, distortion)
    res = distort.perform_operation([img])
    img = res[0]
    return img

class LossTracker(object):
    """Tracks model loss for augmentation scheduling."""
    def __init__(self, value=10):
        self.value = value
        return
        
    def set_value(self, value):
        if value:
            self.value = value
        return
        
    def __float__(self):
        return float(self.value)

def train_gen(df, labels, batch_size=20, path='./train/', augmentation=True, loss_obj=1):
    """Produces a random batch of training examples."""
    count = 0
    while True:
        count += 1
        batch = []
        labels = []
        while len(batch) < batch_size:
            #add tensor to batch
            idx = df.sample().index[0]
            img = image_from_index(idx, df, path)
            #augment image
            loss = float(loss_obj)
            if augmentation:
                img = augment(img, loss=loss)
            tensor = as_tensor(img)
            batch.append(tensor)
            #create one-hot encoded label vector
            label = df.loc[idx]['label']
            labels_code = df.label.unique()
            labels_code.sort()
            vector = np.where(labels_code == label, 1, 0)
            labels.append(vector)
        X = np.array(batch)
        Y = np.array(labels)
        yield X, Y

def val_gen(df, labels, batch_size=None, path='./eval/'):
    """Produces a random batch of training examples."""
    if batch_size==None:
        batch_size = len(labels)*19
    start_batch = 0
    end_batch = start_batch + batch_size
    while True:
        batch = []
        labels = []
        for n in range(start_batch, end_batch):
            #add tensor to batch
            idx = n
            img = image_from_index(idx, df, path)
            tensor = as_tensor(img)
            batch.append(tensor)
            #create one-hot encoded label vector
            label = df.loc[idx]['label']
            labels_code = df.label.unique()
            labels_code.sort()
            vector = np.where(labels_code == label, 1, 0)
            labels.append(vector)
        start_batch += batch_size
        end_batch += batch_size
        if end_batch > len(df):
            start_batch = 0
            end_batch = start_batch + batch_size
        X = np.array(batch)
        Y = np.array(labels)
        yield X, Y


def quiz_batch(indices, df, labels, path='./eval/'):
    """Makes a batch for quizzing. Returns data
    for given indices."""
    batch = []
    labels = []
    for idx in indices:
        #add tensor to batch
        img = image_from_index(idx, df, path)
        tensor = as_tensor(img)
        batch.append(tensor)
        #create one-hot encoded label vector
        label = df.loc[idx]['label']
        labels_code = df.label.unique()
        labels_code.sort()
        vector = np.where(labels_code == label, 1, 0)
        labels.append(vector)
    X = np.array(batch)
    Y = np.array(labels)
    return X, Y
        
def quiz(model, df, labels, path='./eval/', verbose=1):
    """Runs the 'quiz' used in the paper:
    Given 20 choices, choose the character belonging
    to the correct class.
    Because this is a COMPUTER, we'll reframe the test:
    1. Choose 20 characters, one from each class.
    2. For each class label, assign the max value to that class.
    3. Return classwise accuracy.
    """
    correct = 0
    total = 0
    classified = 0
    total_classified = 0
    for n in range(2, 21):
        test_no = str(n)
        if len(test_no) == 1:
            test_no = '0' + test_no
        test_samples = df[['_' + test_no in x for x in df.file]]
        X, Y = quiz_batch(test_samples.index, df=df, labels=labels)
        prediction = model.predict(X)
        correct_answers = np.argmax(Y, axis=1)
        predicted_answers = np.argmax(prediction, axis=1)
        num_correct = np.where(correct_answers == predicted_answers, 1, 0).sum()
        correct += num_correct
        total += len(correct_answers)
        if verbose==2:
            print(
                'quiz no.', 
                test_no, 
                ' results: {}/{} correct.'.format(
                    num_correct, 
                    len(correct_answers)
                ))
        correct_classifications = np.argmax(Y, axis=0)
        classifications = np.argmax(prediction, axis=0)
        num_classified = np.where(correct_classifications == classifications, 1, 0).sum()
        classified += num_classified
        total_classified += len(correct_classifications)
        if verbose==2:
            print(
                'quiz no.', 
                test_no, 
                ' classification results: {}/{} correct.'.format(
                    num_classified, 
                    len(correct_classifications)
                ))
    if verbose==1:
        print('Quiz results: {} out of {} correct.'.format(correct, total))
        print('Quiz accuracy score: {}. Error rate: {}'.format(
            round(correct/total, 3), 
            round(1-correct/total, 3))
             )
        print('Classification results: {} out of {} correct.'.format(classified, total_classified))
        print('Classification accuracy score: {}. Error rate: {}'.format(
            round(classified/total_classified, 3), 
            round(1-classified/total_classified, 3))
             )
    return correct, total, classified, total_classified

def quiz_models(directory, test, labels, capsnet=False):
    """Loads models from directory and runs the quiz on them."""
    model_directory = './models' + directory[1:]
    print('\nquizzing best accuracy model...\n')
    best_acc = load_model(model_directory + 'best_acc.h5')
    if capsnet:
        best_acc = load_model(model_directory + 'best_acc_caps.h5')
    results1 = quiz(best_acc, test, labels)
    print('\nquizzing best loss model...\n')
    best_loss = load_model(model_directory + 'best_loss.h5')
    if capsnet:
        best_acc = load_model(model_directory + 'best_loss_caps.h5')
    results2 = quiz(best_loss, test, labels)
    print('\nquizzing best overfit model...\n')
    overfit = load_model(model_directory + 'overfit.h5')
    if capsnet:
        best_acc = load_model(model_directory + 'overfit_caps.h5')
    results3 = quiz(overfit, test, labels)
    return results1, results2, results3
