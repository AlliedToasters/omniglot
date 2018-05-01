import Augmentor
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import load_model
from skimage.transform import SimilarityTransform, AffineTransform, ProjectiveTransform, warp, rotate

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
    if type(image) == type(np.array([])):
        as_array = image
    else:
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

def trapezoid_shift(image, magnitude):
    mag = magnitude // 2
    a = random.randint(-mag, mag)
    b = random.randint(-mag, mag)
    c = random.randint(-mag, mag)
    d = random.randint(-mag, mag)
    src = np.array([[0, 0], [0, 105], [105, 105], [105, 0]])
    dst = np.array([[0 + a, 0 + b], [0 + c, 105 - b], [105 - c, 105 + d], [105 - a, 0 - d]])

    proj_tform = ProjectiveTransform()
    proj_tform.estimate(src, dst)
    warped = warp(image, proj_tform, output_shape=(105, 105), mode='edge')
    return warped

def distort(image, magnitude=0):
    """Uses PIL, so must be PIL object."""
    a = max(0, magnitude-4)
    distortion = random.randint(0, a)
    dstrt = Augmentor.Operations.Distort(1, 4, 4, distortion)
    res = dstrt.perform_operation([image])
    return res[0]
    
def rotate_transform(image, magnitude):
    """Applies rotation transform."""
    a = magnitude
    rot = random.randint(-a, a)
    arr = rotate(image, angle=rot, mode='edge')
    return arr
    
def nudge_transform(image, magnitude):
    """Nudges image by some number of pixels."""
    a = min(8, (magnitude + 2))
    tx = random.randint(-a, a)
    ty = random.randint(-a, a)
    tform = SimilarityTransform(translation=(ty, tx))
    arr = warp(image, tform, mode='edge')
    return arr

def augment(img, loss=2):
    """applies a series of augmentations to 
    input image and returns.
    """
    max_magnitude = int(max(1.5 - loss, 0) * 10)
    if max_magnitude == 0:
        magnitude = 0
    else:
        magnitude = random.randint(0, max_magnitude)
    img = distort(img, magnitude)
    #convert to array for these transforms.
    arr = np.array(img)
    arr = rotate_transform(arr, magnitude)
    arr = nudge_transform(arr, magnitude)
    arr = trapezoid_shift(arr, magnitude)
    return arr

class LossTracker(object):
    """Tracks model loss for augmentation scheduling."""
    def __init__(self, value=10, scale_by=1):
        self.value = value
        self.scale = scale_by
        return
        
    def set_value(self, value):
        if value:
            self.value = value
        return
        
    def __float__(self):
        return float(self.value * self.scale)

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
        
def get_train_img(idx, df):
    """Shortcut to display training image."""
    path = './train/'
    img = image_from_index(idx, df, path)
    return img

def capsnet_visualize_quiz(X, Y, prediction, quiz_labels, train_df, reconstruction):
    """Visualizes the results of a 'quiz'.
    for presentation.
    """
    quiz_Y = Y[:, quiz_labels]
    quiz_prediction = prediction[:, quiz_labels]
    correct_answers = quiz_labels[np.argmax(quiz_Y, axis=0)]
    predicted_answers = quiz_labels[np.argmax(quiz_prediction, axis=0)]
    correct_idx = np.argwhere(correct_answers == predicted_answers)[0][0]
    correct_class = quiz_labels[correct_idx]
    incorrect_idx = np.argwhere(correct_answers != predicted_answers)[0][0]
    incorrect_class = quiz_labels[incorrect_idx]
    confused_class = predicted_answers[incorrect_idx]
    #Correct answer example
    train_img = get_train_img(correct_class, train_df)
    fig, ax = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle('Correct Result')
    current_label = 0
    for i in range(5):
        for j in range(5):
            if i == 0:
                if j == 1:
                    ax[i, j].set_title('training image')
                    ax[i, j].imshow(train_img, cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                elif j == 3:
                    ax[i, j].set_title('reconstructed image')
                    ax[i, j].imshow(reconstruction[correct_idx, :, :, 0], cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                else:
                    ax[i, j].set_visible(False)
            else:
                likelihood = quiz_prediction[current_label, correct_idx]
                img = X[current_label, :, :, 0]
                title = '{}'.format(str(round(likelihood, 5)))
                if quiz_Y[current_label, correct_idx] == 1:
                    title += ('(correct)')
                title = '{}'.format(title)
                ax[i, j].set_title(title)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].imshow(img, cmap='Greys_r')
                current_label += 1
    plt.show()
    
    #Incorrect answer example
    train_img = get_train_img(incorrect_class, train_df)
    fig, ax = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle('Incorrect Result')
    current_label = 0
    for i in range(5):
        for j in range(5):
            if i == 0:
                if j == 1:
                    ax[i, j].set_title('training image')
                    ax[i, j].imshow(train_img, cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                if j == 3:
                    ax[i, j].set_title('confused reconstruction')
                    ax[i, j].imshow(reconstruction[incorrect_idx, :, :, 0], cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                else:
                    ax[i, j].set_visible(False)
            else:
                likelihood = quiz_prediction[current_label, incorrect_idx]
                img = X[current_label, :, :, 0]
                title = '{}'.format(str(round(likelihood, 5)))
                if quiz_Y[current_label, incorrect_idx] == 1:
                    title += ('(correct)')
                elif quiz_labels[current_label] == confused_class:
                    title += ('(selected)')
                title = '{}'.format(title)
                ax[i, j].set_title(title)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].imshow(img, cmap='Greys_r')
                current_label += 1
    plt.show()
    return

def visualize_quiz(X, Y, prediction, quiz_labels, train_df):
    """Visualizes the results of a 'quiz'.
    for presentation.
    """
    quiz_Y = Y[:, quiz_labels]
    quiz_prediction = prediction[:, quiz_labels]
    correct_answers = quiz_labels[np.argmax(quiz_Y, axis=0)]
    predicted_answers = quiz_labels[np.argmax(quiz_prediction, axis=0)]
    correct_idx = np.argwhere(correct_answers == predicted_answers)[0][0]
    correct_class = quiz_labels[correct_idx]
    incorrect_idx = np.argwhere(correct_answers != predicted_answers)[0][0]
    incorrect_class = quiz_labels[incorrect_idx]
    confused_class = predicted_answers[incorrect_idx]
    #Correct answer example
    train_img = get_train_img(correct_class, train_df)
    fig, ax = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle('Correct Result')
    current_label = 0
    for i in range(5):
        for j in range(5):
            if i == 0:
                if j == 2:
                    ax[i, j].set_title('training image')
                    ax[i, j].imshow(train_img, cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                else:
                    ax[i, j].set_visible(False)
            else:
                likelihood = quiz_prediction[current_label, correct_idx]
                img = X[current_label, :, :, 0]
                title = '{}'.format(str(round(likelihood, 5)))
                if quiz_Y[current_label, correct_idx] == 1:
                    title += ('(correct)')
                title = '{}'.format(title)
                ax[i, j].set_title(title)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].imshow(img, cmap='Greys_r')
                current_label += 1
    plt.show()
    
    #Incorrect answer example
    train_img = get_train_img(incorrect_class, train_df)
    fig, ax = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle('Incorrect Result')
    current_label = 0
    for i in range(5):
        for j in range(5):
            if i == 0:
                if j == 2:
                    ax[i, j].set_title('training image')
                    ax[i, j].imshow(train_img, cmap='Greys_r')
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])
                else:
                    ax[i, j].set_visible(False)
            else:
                likelihood = quiz_prediction[current_label, incorrect_idx]
                img = X[current_label, :, :, 0]
                title = '{}'.format(str(round(likelihood, 5)))
                if quiz_Y[current_label, incorrect_idx] == 1:
                    title += ('(correct)')
                elif quiz_labels[current_label] == confused_class:
                    title += ('(selected)')
                title = '{}'.format(title)
                ax[i, j].set_title(title)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].imshow(img, cmap='Greys_r')
                current_label += 1
    plt.show()
    return
    
    

def quiz(model, df, labels, path='./eval/', verbose=1, visualize=False, train_df=None, capsnet=False):
    """Runs the 'quiz' used in the paper:
    Given 20 choices, choose the character belonging
    to the correct class.
    Because this is a COMPUTER, we'll reframe the test:
    1. Choose 20 characters, one from each class.
    2. For each class label, assign the max value to that class.
    3. Return classwise accuracy.
    """
    visualized = False
    correct = 0
    total = 0
    classified = 0
    total_classified = 0
    for n in range(2, 21):
        test_no = str(n)
        if len(test_no) == 1:
            test_no = '0' + test_no
        test_samples = df[['_' + test_no in x for x in df.file]].sample(20, replace=False)
        quiz_labels = test_samples.label.values.astype(int)
        X, Y = quiz_batch(test_samples.index, df=df, labels=labels)
        quiz_Y = Y[:, quiz_labels]
        prediction = model.predict(X)
        if capsnet:
            prediction, reconstructed = prediction
        quiz_prediction = prediction[:, quiz_labels]
        correct_answers = quiz_labels[np.argmax(quiz_Y, axis=0)]
        predicted_answers = quiz_labels[np.argmax(quiz_prediction, axis=0)]
        num_correct = np.where(correct_answers == predicted_answers, 1, 0).sum()
        if visualize and not visualized:
            if 0 < num_correct < 20:
                if capsnet:
                    capsnet_visualize_quiz(X, Y, prediction, quiz_labels, train_df=train_df, reconstruction=reconstructed)
                    visualized = True
                else:
                    visualize_quiz(X, Y, prediction, quiz_labels, train_df=train_df)
                    visualized = True
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
        correct_classifications = np.argmax(Y, axis=1)
        classifications = np.argmax(prediction, axis=1)
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

def plot_training_characters(train):
    """Plots the training characters for presentation"""
    path = './train/'
    images = train.iterrows()
    n_rows = len(train)//7 + 1
    fig, ax = plt.subplots(n_rows, 7, figsize=(12, n_rows*1.5));
    for i, row in enumerate(ax):
        for j, column in enumerate(row):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            try:
                k, next_character = next(images)
            except:
                ax[i, j].set_visible(False)
            label = next_character.label
            file = next_character.file
            img = np.array(Image.open(path + file))
            ax[i, j].set_title('Label {}'.format(label+1))
            ax[i, j].imshow(img, cmap = 'Greys_r')
    plt.tight_layout()
    plt.show();
    return

def quiz_models(directory, test, labels, visualize=False, capsnet=False, eval_model=None, train_df=None):
    """Loads models from directory and runs the quiz on them.
    If capsnet, eval_model must be passed in."""
    model_directory = './models' + directory[1:]
    print('\nquizzing best accuracy model...\n')
    if capsnet:
        best_acc = eval_model.load_weights(model_directory + 'best_acc_caps.h5')
    else:
        best_acc = load_model(model_directory + 'best_acc.h5')
    results1 = quiz(best_acc, test, labels)
    print('\nquizzing best loss model...\n')
    if capsnet:
        best_loss = eval_model.load_weights(model_directory + 'best_loss_caps.h5')
    else:
        best_loss = load_model(model_directory + 'best_loss.h5')
    results2 = quiz(best_loss, test, labels, visualize=visualize, train_df=train_df)
    print('\nquizzing best overfit model...\n')
    if capsnet:
        overfit = eval_model.load_weights(model_directory + 'overfit_caps.h5')
    else:
        overfit = load_model(model_directory + 'overfit.h5')
    results3 = quiz(overfit, test, labels)
    return results1, results2, results3
