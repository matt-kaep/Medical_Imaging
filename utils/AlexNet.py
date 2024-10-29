from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
import cv2 as cv
import os
import DarkArtefactRemoval as dca
import dullrazor as dr
import utils.segmentation_and_preprocessing as sp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import gradcam_test as gc
#import vanilla_backprop as vb
import compute_features as cf
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, \
                            recall_score, accuracy_score, classification_report


def AlexNet(num_classes):

    """
    This function defines the architecture of the AlexNet model.
    :param num_classes: The number of classes for the output layer.
    :return: The compiled AlexNet model.
    """

    #Define the model
    model = Sequential()

    # C1 Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(227, 227, 3), kernel_size=(11, 11),
                     strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalization
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalization
    model.add(BatchNormalization())

    # C4 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalization
    model.add(BatchNormalization())

    # C5 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalization
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    # D1 Dense Layer
    model.add(Dense(4096, input_shape=(227 * 227 * 3,)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D3 Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


class Metrics(Callback):

    """
    This class is a custom callback for Keras that calculates and prints validation metrics after each epoch.
    """

    def __init__(self, valid_generator=None):
        """
        Initialize the Metrics object.
        :param valid_generator: The validation data generator.
        """
        self.X_test, self.Y_test = None, None
        self.valid_iterator = valid_generator

    def on_train_begin(self, logs={}):
        """
        Initialize lists to store validation metrics.
        :param logs: Dictionary of logs.
        """
        self.accuracy = []
        self.f1_scores = []
        self.recalls = []
        self.precisions = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate and print validation metrics after each epoch.
        :param epoch: The current epoch.
        :param logs: Dictionary of logs.
        """
        self.X_test, self.Y_test = self.valid_iterator.next()
        pred = (np.asarray(self.model.predict(self.X_test))).round()
        predicted_label = np.argmax(pred, axis=1)

        target = self.Y_test
        target_label = np.argmax(target, axis=1)

        val_accuracy = accuracy_score(target_label, predicted_label)
        val_f1 = f1_score(target_label, predicted_label, average='weighted')
        val_recall = recall_score(target_label, predicted_label, average='weighted')
        val_precision = precision_score(target_label, predicted_label, average='weighted')

        self.accuracy.append(val_accuracy)
        self.f1_scores.append(val_f1)
        self.recalls.append(val_recall)
        self.precisions.append(val_precision)

        # print(confusion_matrix(target_label, predicted_label))
        # print(classification_report(target_label, predicted_label))

        print("Validation:")
        print("acc: {a:.2f} - f1-score: {f1:.2f} - prec: {p:.2f} - recall {r:.2f}"
              .format(a=val_accuracy, f1=val_f1, p=val_precision, r=val_recall))
        

def plot_history(history, metrics):

    """
    This function plots the training and validation accuracy and loss after each epoch.
    :param history: The history object returned by model.fit().
    :param metrics: The Metrics object.
    """

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    # plot loss for each epoch
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.3f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # plot accuracy for each epoch
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.3f'))+')')   
    
    plt.plot(epochs, metrics.accuracy, 'r', label='Validation Accuracy (' + str(format(np.mean(metrics.accuracy),'.3f'))+')')
    
    plt.title('Accuracy Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # plot all mean performance metrics (accuracy, precision, recall and F1-score) for each epoch
    plt.figure(3)
    plt.plot(epochs, metrics.accuracy, 'r', label='Accuracy (' + str(format(np.mean(metrics.accuracy),'.3f'))+')')
    plt.plot(epochs, metrics.precisions, 'g', label='Precision (' + str(format(np.mean(metrics.precisions),'.3f'))+')')
    plt.plot(epochs, metrics.recalls, 'b', label='Recall (' + str(format(np.mean(metrics.recalls),'.3f'))+')')
    plt.plot(epochs, metrics.f1_scores, 'y', label='F1-score (' + str(format(np.mean(metrics.f1_scores),'.3f'))+')')

    plt.title('Validation set performances')
    plt.xlabel('Epochs')
    plt.ylabel('Performance metric')
    plt.legend()
    
    plt.show();


def plot_confusion_matrix(true_labels, predicted_labels, class_labels):

    """
    This function plots the confusion matrix for the test set.
    :param true_labels: The true labels for the test set.
    :param predicted_labels: The predicted labels for the test set.
    :param class_labels: The labels for each class.
    """

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure()
    plt.title('Confusion matrix')
    sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=sorted(class_labels), yticklabels=sorted(class_labels))
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.draw()
    plt.tight_layout()
    plt.show();

#import accuracy_score, f1_score, recall_score, precision_score, classification_report from keras


def run_alexnet(train_dataframe, val_dataframe, test_dataframe, class_labels, dir_path, batch_size, rand_seed, num_epochs, weights_filename, gpu_index, adv_preproc=False):

    """
    This function trains the AlexNet model and evaluates it on the test set.
    :param train_dataframe: The dataframe for the training set.
    :param val_dataframe: The dataframe for the validation set.
    :param test_dataframe: The dataframe for the test set.
    :param class_labels: The labels for each class.
    :param dir_path: The directory path for the images.
    :param batch_size: The batch size for training.
    :param rand_seed: The random seed for reproducibility.
    :param num_epochs: The number of epochs for training.
    :param weights_filename: The filename to save the model weights.
    :param gpu_index: The index of the GPU to use.
    :param adv_preproc: Whether to use advanced preprocessing for the training set.
    """

    print('{:-<50}'.format(""))
    print("Image classification using AlexNet CNN classifier:")



    # Configurer TensorFlow pour utiliser le GPU spécifié
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        except RuntimeError as e:
            print(e)


    if class_labels is None:
        print("Error: 'class_labels' cannot be None")
        return
    
    if train_dataframe is None:
        print("Error: 'train_dataframe' cannot be None")
        return
    
    if test_dataframe is None:
        print("Error: 'test_dataframe' cannot be None")
        return

    num_samples_train = train_dataframe['ID'].count()
    
    if adv_preproc:
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           # featurewise_std_normalization=True,
                                           rotation_range=180,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           horizontal_flip=True,
                                           zoom_range=0.15,
                                           brightness_range=[0.8, 1.2],
                                           data_format='channels_last')
    else:
        # Use ImageDataGenerator class to build an image generator for the training set.
        # It will also rescale the pixel values between 0 and 1, that is by multiplying
        # them by a factor of 1/255 since our original images consist in RGB coefficients
        # in [0, 255], but such values would be too high for our model to process.
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           data_format='channels_last')
        
    train_iterator = train_datagen.flow_from_dataframe(train_dataframe,
                                                       directory=dir_path,
                                                       x_col='ID',
                                                       y_col='CLASS',
                                                       target_size=(227, 227),
                                                       batch_size=batch_size,
                                                       color_mode='rgb',
                                                       class_mode='categorical',
                                                       shuffle=True,
                                                       seed=rand_seed)

    # Use ImageDataGenerator class to build an image generator to be used for model validation.
    # It will use the test set images as input data
    valid_datagen = ImageDataGenerator(rescale=1. / 255,
                                       data_format='channels_last')
        
    validation_iterator = valid_datagen.flow_from_dataframe(val_dataframe,
                                                            directory=dir_path,
                                                            x_col='ID',
                                                            y_col='CLASS',
                                                            target_size=(227, 227),
                                                            batch_size=batch_size,
                                                            color_mode='rgb',
                                                            class_mode='categorical',
                                                            shuffle=True,
                                                            seed=rand_seed)
        
    # Use ImageDataGenerator class to build an image generator to be used for
    # performance evaluation.
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      data_format='channels_last')

    test_iterator = test_datagen.flow_from_dataframe(test_dataframe,
                                                     directory=dir_path,
                                                     x_col='ID',
                                                     y_col='CLASS',
                                                     target_size=(227, 227),
                                                     batch_size=1,
                                                     class_mode='categorical',
                                                     shuffle=False)
    # create a Metrics instance for validation
    metrics = Metrics(validation_iterator)

    # build AlexNet model
    model = AlexNet(len(class_labels))

    # print model summary
    model.summary()

    # train the AlexNet CNN
    try:
        history = model.fit_generator(train_iterator,
                                      epochs=num_epochs,
                                      steps_per_epoch=num_samples_train // batch_size,
                                      verbose=1,
                                      callbacks=[metrics])
    except KeyboardInterrupt:
        pass

    model.save(weights_filename)
    print('\nModel weights saved successfully on file {name}\n'.format(name=weights_filename))

    # print and plot mean values of performance metrics related to the Validation set
    print('*** VALIDATION SET PERFORMANCE EVALUATION ***')
    print('Mean accuracy: {:.3f}'.format(np.mean(metrics.accuracy)))
    print('Mean precision: {:.3f}'.format(np.mean(metrics.precisions)))
    print('Mean recall: {:.3f}'.format(np.mean(metrics.recalls)))
    print('Mean f1-score: {:.3f}'.format(np.mean(metrics.f1_scores)))
    
    plot_history(history, metrics)

    # *** TEST SET PERFORMANCE EVALUATION ***
    # get prediction on test data
    y_pred = model.predict_generator(test_iterator, steps=len(test_iterator), verbose=1)
    y_pred = np.argmax(y_pred, axis=1)

    # compute and plot performance metrics values for each class
    accuracy = accuracy_score(test_iterator.classes, y_pred)
    val_f1 = f1_score(test_iterator.classes, y_pred, average='weighted')
    val_recall = recall_score(test_iterator.classes, y_pred, average='weighted')
    val_precision = precision_score(test_iterator.classes, y_pred, average='weighted')

    print('*** TEST SET PERFORMANCE EVALUATION - AlexNet CNN ***')
    print('Accuracy: {:.3f}'.format(accuracy))
    print('F1-score: {:.3f}'.format(val_f1))
    print('Recall: {:.3f}'.format(val_recall))
    print('Precision: {:.3f}'.format(val_precision))

    # print classification report and plot confusion matrix
    print('\nClassification Report')
    print( classification_report(test_iterator.classes, y_pred))

    plot_confusion_matrix(test_iterator.classes, y_pred, class_labels)