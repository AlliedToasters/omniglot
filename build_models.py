import numpy as np
import os
import matplotlib.pyplot as plt
from capsulenet import CapsNet, MarginLoss
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Reshape, Add, Flatten
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, TensorBoard
from keras.optimizers import Adam, SGD

class LossLoggerCNN(Callback):
    def __init__(self, logger):
        """Give it a logger"""
        self.logger = logger
        
    def on_batch_end(self, batch, logs={}):
        self.logger.set_value(logs.get('loss'))
        
class LossLoggerCaps(Callback):
    def __init__(self, logger):
        """Give it a logger"""
        self.logger = logger
        
    def on_batch_end(self, batch, logs={}):
        self.logger.set_value(logs.get('loss'))

def make_convnet(input_shape, n_class, width=2, dropout=.2):
    """Builds a classic CNN model. width adjusts layer width
    for easy adjustment of total number of parameters.
    Returns a compiled model."""
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=1*width, kernel_size=9, padding='valid', activation='relu', name='conv1')(inputs)
    x = Conv2D(filters=2*width, kernel_size=5, padding='valid', activation='relu', name='conv2')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2,2), name='maxp1')(x)
    x = Conv2D(filters=2*width, kernel_size=2, padding='valid', activation='relu', name='conv3')(x)
    x = Conv2D(filters=4*width, kernel_size=2, padding='valid', activation='relu', name='conv4')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2,2), name='maxp2')(x)
    x = Conv2D(filters=4*width, kernel_size=2, padding='valid', activation='relu', name='conv5')(x)
    x = Conv2D(filters=8*width, kernel_size=2, padding='valid', activation='relu', name='conv6')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2,2), name='maxp3')(x)
    x = Conv2D(filters=8*width, kernel_size=2, padding='valid', activation='relu', name='conv7')(x)
    x = Conv2D(filters=16*width, kernel_size=2, padding='valid', activation='relu', name='conv8')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2,2), name='maxp4')(x)
    x = Conv2D(filters=16*width, kernel_size=2, padding='valid', activation='relu', name='conv9')(x)
    x = Conv2D(filters=32*width, kernel_size=2, padding='valid', activation='relu', name='conv10')(x)
    x = Conv2D(filters=37*width, kernel_size=2, padding='valid', activation='relu', name='conv11')(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(64*width, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(16*width, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64*width, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(n_class, activation='softmax')(x)


    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(decay=.0001, nesterov=True, momentum=.1),
        metrics=['categorical_accuracy']
    )
    return model

def train_convnet(model, train_generator, val_generator, directory, verbose=False, loss_obj=None, epochs=200):
    """Trains models for 100 epochs. Saves best validation accuracy,
    best validation loss, and final "overfit" model, into:
    ./models/ + directory + 'best_acc.h5', 'best_loss.h5',
    'overfit.h5'
    """
    save_path = './models' + directory[1:]
    os.system('mkdir models')
    os.system('mkdir {}'.format('/'.join(save_path.split('/')[:3])))
    os.system('mkdir {}'.format('/'.join(save_path.split('/')[:4])))
    callbacks = []
    acc_ = ModelCheckpoint(save_path + 'best_acc.h5', monitor='val_categorical_accuracy', save_best_only=True, verbose=verbose)
    callbacks.append(acc_)
    loss_ = ModelCheckpoint(save_path + 'best_loss.h5', monitor='val_loss', save_best_only=True, verbose=verbose)
    callbacks.append(loss_)
    tboard = TensorBoard('./logs', batch_size=30)
    callbacks.append(tboard)
    if loss_obj != None:
        loss_logger = LossLoggerCNN(loss_obj)
        callbacks.append(loss_logger)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs=epochs,
        validation_data = val_generator,
        validation_steps = 1,
        callbacks = callbacks, 
        verbose=verbose
    )
    model.save(save_path + 'overfit.h5')
    return history


def make_capsnet(input_shape, n_class, routings, reconstruction_loss, lambda_downweight=.4):
    """
    Puts together a modified capsnet for the problem.
    Returns a compiled model.
    lambda_downweight is the downweight given to negative examples
    to avoid all capsules shrinking to zero. Lower value for higher
    number of classes.
    """
    x = Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = Conv2D(filters=128, kernel_size=11, strides=2, padding='valid', activation='relu', name='conv1')(x)
    conv2 = Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv2')(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction.
    decoder = Sequential(name='decoder')
    decoder.add(Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(Dense(1024, activation='relu'))
    # Added another dense layer for additional pattern learning capacity.
    decoder.add(Dense(1024, activation='relu'))
    decoder.add(Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = Input(shape=(n_class, 16))
    noised_digitcaps = Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = Model([x, y, noise], decoder(masked_noised_y))
    #train_model, eval_model, manipulate_model

    margin_loss = MarginLoss(lambda_downweight=lambda_downweight)
    train_model.compile(optimizer=Adam(lr=.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., reconstruction_loss],
                  metrics={'capsnet': 'categorical_accuracy'}
                       )

    
    
    return train_model, eval_model, manipulate_model

def train_capsnet(model, train_generator, val_generator, directory, 
                  verbose=False, 
                  lr=.001, 
                  lr_decay=.9, 
                  loss_obj=None, 
                  epochs=100,
                  validation_steps=19
                 ):
    """Trains models for 100 epochs. Saves best validation accuracy,
    best validation loss, and final "overfit" model (weights only), into:
    ./models/ + directory + 'best_acc_caps.h5', 'best_loss_caps.h5',
    'overfit.h5'
    """
    save_path = './models' + directory[1:]
    os.system('mkdir models')
    os.system('mkdir {}'.format('/'.join(save_path.split('/')[:3])))
    os.system('mkdir {}'.format('/'.join(save_path.split('/')[:4])))
    callbacks = []
    acc_checkpoint = ModelCheckpoint(
        save_path + 'best_acc_caps.h5', 
        monitor='val_capsnet_categorical_accuracy', 
        save_best_only=True, 
        verbose=verbose, 
        save_weights_only=True
    )
    callbacks.append(acc_checkpoint)
    loss_checkpoint = ModelCheckpoint(
        save_path + 'best_loss_caps.h5', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=verbose, 
        save_weights_only=True
    )
    callbacks.append(loss_checkpoint)
    lr_ = LearningRateScheduler(schedule=lambda epoch: lr * (lr_decay ** epoch))
    callbacks.append(lr_)
    if loss_obj != None:
        loss_logger = LossLoggerCaps(loss_obj)
        callbacks.append(loss_logger)
    
    def caps_tg(gen=train_generator):
        while True:
            X, Y = next(gen)
            yield ([X, Y], [Y, X])

    ctg = caps_tg(train_generator)

    def caps_vg(gen=val_generator):
        while True:
            X, Y = next(gen)
            yield ([X, Y], [Y, X])

    cvg = caps_vg(val_generator)
    
    history = model.fit_generator(
        ctg,
        steps_per_epoch = 100,
        epochs=epochs,
        validation_data = cvg,
        validation_steps = validation_steps,
        callbacks = callbacks, 
        verbose=verbose
    )
    model.save_weights(save_path + 'overfit_caps.h5')
    return history


def plot_history(history, model_name='model', capsnet=False):
    """Takes a history object and makes some plots."""
    if capsnet:
        plt.plot(history.history['capsnet_categorical_accuracy']);
        plt.plot(history.history['val_capsnet_categorical_accuracy']);
    else:
        plt.plot(history.history['categorical_accuracy']);
        plt.plot(history.history['val_categorical_accuracy']);
    plt.title('{} accuracy'.format(model_name));
    plt.ylabel('accuracy');
    plt.xlabel('epoch');
    plt.legend(['train', 'test'], loc='upper left');
    plt.show();
    
    if capsnet:
        plt.plot(history.history['decoder_loss']);
        plt.plot(history.history['val_decoder_loss']);
        plt.title('{} reconstruction loss'.format(model_name));
        plt.ylabel('loss');
        plt.xlabel('epoch');
        plt.legend(['train', 'test'], loc='upper left');
        plt.show();
        
    
    if capsnet:
        plt.plot(history.history['capsnet_loss']);
        plt.plot(history.history['val_capsnet_loss']);
        plt.title('{} classification loss'.format(model_name));
        plt.ylabel('loss');
        plt.xlabel('epoch');
        plt.legend(['train', 'test'], loc='upper left');
        plt.show();

    plt.plot(history.history['loss']);
    plt.plot(history.history['val_loss']);
    if capsnet:
        title = '{} combined loss'.format(model_name)
    else:
        title = '{} loss'.format(model_name)
    plt.title(title);
    plt.ylabel('loss');
    plt.xlabel('epoch');
    plt.legend(['train', 'test'], loc='upper left');
    plt.show();
    return