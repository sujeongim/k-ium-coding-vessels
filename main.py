from keras.applications import ResNet50
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Input, LSTM, Dense, TimeDistributed, Add, Reshape, Multiply, Activation, Lambda
from utils.SpatialAttention import SpatialAttentionLayer
import tensorflow as tf
import logging
import time
import os
import pandas as pd
from utils.ImagePreprocess import preprocess, crop
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from model import build_model
import argparse

  
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate a baseline model for the dataset.')
    parser.add_argument('--test_path', type=str, default=None, help='Path to the CSV file directory.')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt', help='Path to the model checkpoint directory.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--positive_only', action='store_true', help='Use only positive samples for evaluation.')
    parser.add_argument('--output_file', type=str, default='./results/output.csv', help='Path to save the output CSV file.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for converting probabilities to binary predictions.')
    config = parser.parse_args()
    
    
    # CSV_PATH = "" ####YOUR .CSV DIR HERE####
    CSV_FILENAME = "test.csv" ####YOUR .CSV DIR HERE####
    IMG_PATH = f'{config.test_path}/images' ####YOUR TEST IMAGE FILES DIR HERE####
    IMG_FILE_EXTENSION = "*.jpg" ####YOUR TEST IMAGE FILE EXTENSION HERE####

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = build_model(input_shape=(224, 224, 3), num_labels=22)
    dummy_input = np.random.random((1, 8, 224, 224, 3))
    _ = model(dummy_input)
    
    # model.build(input_shape=(None, 8, 224, 224, 3))
    
    ### Loading Checkpoints ###
    logger.info("Loading the latest checkpoint...")
    model.load_weights(config.ckpt_path)


    
    ### Loading test set ###
    logger.info("Loading test.csv...")
    
    test = pd.read_csv(os.path.join(config.test_path, CSV_FILENAME))
    if config.positive_only:
        test = test[test['Aneurysm'].astype(int) == 1]
        indices = test['Index'].to_numpy(dtype='int32')
        logger.info(f"Filtered to {len(test)} positive samples.")
    
    logger.info("Loading test images...")
    imgfiles = sorted(glob.glob(os.path.join(IMG_PATH, IMG_FILE_EXTENSION))) 
    if config.positive_only:
        imgfiles = [f for f in imgfiles if int(f.split('/')[-1][1:4]) in indices]
        # assert all(int(f.split('/')[-1][1:4]) == idx for f, idx in zip(imgfiles, indices)), f"Image files and indices do not match elementwise: {imgfiles[:10]} vs {indices[:10]}"
        logger.info(f"Filtered to {len(imgfiles)} images corresponding to positive samples.")
    # print(imgfiles)
    
    images = []
    smallest_size = (224, 224)
    
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
    test_img = np.array(images)
    
    logger.info("Done.")
    
    ### Model prediction and Save output.csv ###
    logger.info("Yielding model prediction...")
    pred = model.predict(test_img, batch_size=config.batch_size)
    
    cols = test.drop(['Index'], axis=1).columns
    # Set a threshold for converting probabilities to binary predictions

    # Apply the threshold to the predictions for binary labels
    binary_predictions = np.where(pred > config.threshold, 1, 0)

    # Get the probabilities for the 'Aneurysm' column
    aneurysm_probabilities = pred[:, 0]  # Assuming 'Aneurysm' is the first column

    # Clip the 'Aneurysm' probabilities between 0 and 1
    aneurysm_probabilities = np.clip(aneurysm_probabilities, 0, 1)

    # Combine the binary predictions with 'Aneurysm' probabilities
    combined_predictions = np.concatenate((aneurysm_probabilities[:, np.newaxis], binary_predictions[:, 1:]), axis=1)

    # Assuming combined_predictions is a NumPy array with shape (num_samples, num_labels)

    # Convert combined_predictions to a DataFrame
    combined_predictions_df = pd.DataFrame(combined_predictions, columns=cols)
    # assuming every column except the one we’re about to add is a 0/1 prediction
    # combined_predictions_df['Aneurysm'] = combined_predictions_df.any(axis=1).astype(int)
    
    combined_predictions_df['Index'] = test['Index'].copy()
    
    logger.info("Saving output.csv...")
    
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    combined_predictions_df.to_csv(config.output_file, index=False)
