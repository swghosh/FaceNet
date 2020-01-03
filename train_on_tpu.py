#!/usr/bin/env python3

"""
We'll use tf.keras for
# training the FaceNet
"""
import tensorflow as tf
from tensorflow import keras

"""
Certain constants are to 
be defined
"""
IMAGE_SIZE = (224, 224)
CHANNELS = 3

TPU_WORKER = 'grpc://10.0.0.1:8470'

BATCH_SIZE = 4096
LEARN_RATE = 0.05
ALPHA = 0.2
EPOCHS = 275

CL_PATH = 'gs://bucket-name/C-MS-Celeb-class-labels.txt'
DATASET_PATH = 'gs://bucket-name/C-MS-Celeb-TFRecords'
TB_PATH = 'gs://bucket-name/C-MS-Celeb-FaceNet_TensorBoard'

"""
Initialise the TPU
and create the required
tf.distribute.Strategy
"""
keras.backend.clear_session()

tpu_cluster = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
tf.config.experimental_connect_to_cluster(tpu_cluster)
tf.tpu.experimental.initialize_tpu_system(tpu_cluster)
strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster)

"""
Prepare the data pipeline
for train, val images
"""
from facenet import dataset
train, val = dataset.get_train_val_dataset(CL_PATH, DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)
# these are essential values that have to be set
# in order to determine the right number of steps per epoch
train_samples, val_samples = 5992411, 471607
# this value is set so as to ensure
#  proper shuffling of dataset
dataset.SHUFFLE_BUFFER = train_samples

"""
Add some tf.keras.callbacks.Callback(s)
to enhance op(s)
like TensorBoard visualisation
and learning rate Scheduling
"""

def lr_schedule(epoch):
    return LEARN_RATE * (0.1 ** (epoch // 100))

lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)
tensorboard = keras.callbacks.TensorBoard(TB_PATH)
checkpoints = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}_{val_loss:.4f}.hdf5',
    monitor='val_loss', save_weights_only=True)

cbs = [lr_scheduler, checkpoints, tensorboard]

"""
Construct the FaceNet NN2
network with TPU strategy
"""
from facenet import facenet
with strategy.scope():
    model = facenet.create_facenet_nn2(IMAGE_SIZE, CHANNELS, ALPHA, LEARN_RATE)

"""
Train the model
"""
train_history = model.fit(train.data, steps_per_epoch=train_samples // BATCH_SIZE + 1,
    validation_data=val.data, validation_steps=val_samples // BATCH_SIZE + 1,
    callbacks=cbs, epochs=EPOCHS)
model.save('model.h5')

"""
Let's visualise how the
training went
"""
from matplotlib import pyplot as plt
def save_plots():
    """
    Save plot 
    Loss vs Epochs
    """

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']

    plt.figure(figsize=(10, 8))

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.ylabel('Triplet Loss')
    plt.xlabel('Epochs')
    plt.title('Loss')

    plt.savefig('epcoch_wise_loss.png')

save_plots()
