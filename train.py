# Script for training custom VAD model for the voxseg toolkit

import voxseg
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras import utils, models, layers
from tensorflow.keras.callbacks import ModelCheckpoint

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10,inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(config=session_conf)

# Define Model
def cnn_bilstm(output_layer_width):
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='elu'), input_shape=(None, 32, 32, 1)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dense(128, activation='elu')))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(output_layer_width, activation='softmax')))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

# Define training parameters
def train_model(model, x_train, y_train, validation_split, x_dev=None, y_dev=None, epochs=25, batch_size=64, callbacks=None):
    if validation_split:
        return model.fit(x_train[:,:,:,:,np.newaxis], y_train, validation_split = validation_split,
                     epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    elif x_dev is not None:
        return model.fit(x_train[:,:,:,:,np.newaxis], y_train,
                     validation_data=(x_dev[:,:,:,:,np.newaxis], y_dev),
                     epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    else:
        print('WARNING: no validation data, or validation split provided.')
        return model.fit(x_train[:,:,:,:,np.newaxis], y_train,
                     epochs=epochs, batch_size=batch_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CHPC_VAD_train.py',
                                     description='Train an instance of the voxseg VAD model.')

    parser.add_argument('-v', '--validation_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', \'utt2spk\' and \'segments\'')

    parser.add_argument('-s', '--validation_split', type=float,
                        help='a percetage of the training data to be used as a validation set, if an explicit validation \
                              set is not defined using -v')

    parser.add_argument('train_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', \'utt2spk\' and \'segments\'')

    parser.add_argument('model_name', type=str,
                        help='a filename for the model, the model will be saved as <model_name>.h5 in the output directory')

    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where the model will be saved as <model_name>.h5')

    args = parser.parse_args()

    # Fetch data
    data_train = voxseg.prep_labels.prep_data(args.train_dir)
    if args.validation_dir:
        data_dev = voxseg.prep_labels.prep_data(args.validation_dir)

    # Extract features
    feats_train = voxseg.extract_feats.extract(data_train)
    feats_train = voxseg.extract_feats.normalize(feats_train)
    if args.validation_dir:
        feats_dev = voxseg.extract_feats.extract(data_dev)
        feats_dev = voxseg.extract_feats.normalize(feats_dev)

    # Extract labels
    labels_train = voxseg.prep_labels.get_labels(data_train)
    labels_train['labels'] = voxseg.prep_labels.one_hot(labels_train['labels'])
    if args.validation_dir:
        labels_dev = voxseg.prep_labels.get_labels(data_dev)
        labels_dev['labels'] = voxseg.prep_labels.one_hot(labels_dev['labels'])

    # Train model
    X = voxseg.utils.time_distribute(np.vstack(feats_train['normalized-features']), 15)
    y = voxseg.utils.time_distribute(np.vstack(labels_train['labels']), 15)
    if args.validation_dir:
        X_dev = voxseg.utils.time_distribute(np.vstack(feats_dev['normalized-features']), 15)
        y_dev = voxseg.utils.time_distribute(np.vstack(labels_dev['labels']), 15)
    else:
        X_dev = None
        y_dev = None

    args.model_name
    checkpoint = ModelCheckpoint(filepath=f'{args.out_dir}/{args.model_name}.h5',
                                 save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True)

    if y.shape[-1] == 2 or y.shape[-1] == 4:
        hist = train_model(cnn_bilstm(y.shape[-1]), X, y, args.validation_split, X_dev, y_dev, callbacks=[checkpoint])

        df = pd.DataFrame(hist.history)
        df.index.name = 'epoch'
        df.to_csv(f'{args.out_dir}/{args.model_name}_training_log.csv')
    else:
        print(f'ERROR: Number of classes {y.shape[-1]} is not equal to 2 or 4, see README for more info on using this training script.')
        