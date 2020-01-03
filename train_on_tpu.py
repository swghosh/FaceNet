#!/usr/bin/env python3

import argparse

argparser = argparse.ArgumentParser(description='Train the FaceNet model on TPU')

argparser.add_argument('--tpu_worker', type=str, 
    help='GRPC connection string for TPU', default='grpc://10.0.0.2:8470')
argparser.add_argument('--batch_size', type=int, 
    help='Batch size for training and validation pipeline', default=8192)
argparser.add_argument('--learning_rate', type=float,
    help='Learning rate for model optimizer', default=0.05)
argparser.add_argument('--image_size', type=int,
    help='Size of W, H dimensions of image', default=8192)
argparser.add_argument('--channels', type=int,
    help='Number of image channels (RGB = 3, Grayscale = 1)', default=3)
argparser.add_argument('--alpha', type=float,
    help='Triplet loss margin', default=0.2)
argparser.add_argument('--epochs', type=float,
    help='Number of training epochs', default=275)
argparser.add_argument('--class_labels', type=str,
    help='Path to file containing list of all class labels', default='gs://bucket-name/C-MS-Celeb-class-labels.txt')
argparser.add_argument('--dataset_path', type=str,
    help='Path to directory containing train and validation TFRecords', default='gs://bucket-name/C-MS-Celeb-TFRecords')
argparser.add_argument('--num_train_samples', type=int,
    help='Number of training images present in dataset', default=5992411)
argparser.add_argument('--num_val_samples', type=int,
    help='Number of validation images present in dataset', default=471607)
argparser.add_argument('--tensorboard_path', type=str,
    help='Directory where TensorBoard logs will be stored', default='gs://bucket-name/FaceNet_TensorBoard')

import tensorflow as tf
from tensorflow import keras

from facenet import facenet
from facenet import dataset

def main(args):
    keras.backend.clear_session()

    tpu_cluster = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_worker)
    tf.config.experimental_connect_to_cluster(tpu_cluster)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster)
    strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster)
    
    train, val = dataset.get_train_val_dataset(args.class_labels, args.dataset_path, (args.image_size, args.image_size), args.batch_size)
    dataset.SHUFFLE_BUFFER = args.num_train_samples
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (0.1 ** (epoch // 100)), verbose=1)
    tensorboard = keras.callbacks.TensorBoard(args.tensorboard_path)
    checkpoints = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}_{val_loss:.4f}.hdf5',
        monitor='val_loss', save_weights_only=True)

    cbs = [lr_scheduler, checkpoints, tensorboard]
    
    with strategy.scope():
        model = facenet.create_facenet_nn2((args.image_size, args.image_size), args.channels, args.alpha, args.learning_rate)
        
    model.fit(train.data, steps_per_epoch=args.num_train_samples // args.batch_size + 1,
        validation_data=val.data, validation_steps=args.num_val_samples // args.batch_size + 1,
        callbacks=cbs, epochs=args.epochs)
    model.save('facenet.h5')

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)