import logging
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.ImagePreprocess import preprocess, crop
from model import build_model
from PIL import Image
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import argparse


if __name__ == '__main__':
    tf.config.list_physical_devices('GPU')
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Detected {len(gpus)} GPU(s):", gpus)
    
    parser = argparse.ArgumentParser(description='Train a baseline model for the dataset.')
    parser.add_argument('--train_dir', type=str, default='', help='Path to the training CSV file directory.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--output_dir', type=str, default='./ckpt', help='Directory to save the model checkpoints.')
    parser.add_argument('--positive_only', action='store_true', help='Use only positive samples for training.')
    config = parser.parse_args()
    
    # TRAIN_PATH = "" ####YOUR .CSV DIR HERE####
    CSV_FILENAME = "train.csv" ####YOUR .CSV DIR HERE####
    IMG_PATH = f'{config.train_dir}/images' ####YOUR TEST IMAGE FILES DIR HERE####
    IMG_FILE_EXTENSION = "*.jpg" ####YOUR TEST IMAGE FILE EXTENSION HERE####
    OUTPUT_PATH = './ckpt' ####YOUR OUTPUT DIR HERE####
    batch_size = 16
    # batch_size = per_gpu_batch * strategy.num_replicas_in_sync
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading train.csv...")
    if config.positive_only:
        train_csv = pd.read_csv(f'{config.train_dir}/{CSV_FILENAME}')
        train_csv = train_csv[train_csv['Aneurysm'].astype(int)==1]
        indices = train_csv['Index'].to_numpy(dtype='int32')
        print(f"Indices of positive samples: {indices[0]}")
        logger.info(f"Filtered to {len(train_csv)} positive samples.")
    else:
        logger.info("Using all samples from train.csv.")
        train_csv = pd.read_csv(f'{config.train_dir}/{CSV_FILENAME}')
    
    labels_by_patient = {}
    for _, row in train_csv.iterrows():
        pid = row['Index']
        label = row[1:].to_numpy(dtype='float32')
        labels_by_patient[pid] = label
    
    # cols = train_csv.columns
    
    logger.info("Loading train images...")
    imgfiles = sorted(glob(f'{IMG_PATH}/{IMG_FILE_EXTENSION}'))
    if config.positive_only:
        imgfiles = [f for f in imgfiles if int(f.split('/')[-1][:4]) in indices]
        logger.info(f"Filtered to {len(imgfiles)} images with positive samples only.")
            
    logger.info(f"Found {len(imgfiles)} images.")
    images = []
    smallest_size = (224, 224)
    
    ### Preprocessing images ###
    for c in tqdm(range(0, len(imgfiles), 8)):
        temp = []
        for filename in imgfiles[c:c+8]:
            image = Image.open(filename)
            pnum = int(filename.split('/')[-1][:4])

            w, h = image.size
            if w != h:
                image = crop(image)

            if image.size != smallest_size:
                image = image.resize(smallest_size)
            image = preprocess(image)
            
            temp.append(np.array(image))
        images.append(np.array(temp))
    # train_img = np.array(images)
    logger.info("Done.")
    
    train_images, val_images, train_labels, val_labels = train_test_split(np.array(images), np.array(list(labels_by_patient.values())), train_size=0.8, random_state=42)
    # val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42)
    
    ### Train a model ###
    model = build_model(input_shape=(224, 224, 3), num_labels=22)
    
    
    n_batches = len(images) / batch_size
    n_batches = math.ceil(n_batches)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=f"{OUTPUT_PATH}/ckpt_{{epoch:02d}}.weights.h5",
                                    save_weights_only=True,
                                    verbose=1,
                                    save_freq=5*n_batches
                                )

    early_stopping = EarlyStopping(monitor = 'val_loss')
    
    init_lr = 1e-5
    epochs = 50
    decay = init_lr / epochs
    
    def time_based_decay(epoch, lr):
        return lr * 1 / (1 + decay * epoch)

    lr_scheduler = LearningRateScheduler(time_based_decay, verbose=1)
    
    # global_batch_size = per_gpu_batch * strategy.num_replicas_in_sync
    logger.info("Train starts...")
    
    model.fit(
            train_images, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data =(val_images, val_labels),
            callbacks=[cp_callback, early_stopping, lr_scheduler]
        )