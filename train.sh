#!/bin/bash

python3 train_on_tpu.py --tpu_worker "grpc://10.0.27.2:8470" \ 
--class_labels="gs://shrill-anstett-us/C-MS-Celeb-class-labels.txt" \ 
--dataset_path="gs://shrill-anstett-us/C-MS-Celeb-TFRecords" \ 
--tensorboard_path="gs://shrill-anstett-us/C-MS-Celeb-FaceNet_TensorBoard" \ 
--num_train_samples=5992411 \ 
--num_val_samples=471607
