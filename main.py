from keras.applications import ResNet50
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential, Model
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

  
if __name__=='__main__':
    CSV_PATH = "./test_set" ####YOUR .CSV DIR HERE####
    CSV_FILENAME = "test.csv" ####YOUR .CSV DIR HERE####
    IMG_PATH = './test' ####YOUR TEST IMAGE FILES DIR HERE####
    IMG_FILE_EXTENSION = "*.jpg" ####YOUR TEST IMAGE FILE EXTENSION HERE####

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    batch_size = 2 # increase/decrease the value according to your RAM storage
    
    ### Building Models ###
    logger.info("Building a model instance...")
    model = build_model(input_shape=(720, 720, 3), num_labels=22)


    ### Loading Checkpoints ###
    logger.info("Loading the latest checkpoint...")
    checkpoint_dir = "./kium-output/"
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    
    ### Loading test set ###
    logger.info("Loading test.csv...")
    
    test = pd.read_csv(os.path.join(CSV_PATH, CSV_FILENAME))
    
    logger.info("Loading test images...")
    imgfiles = sorted(glob.glob(os.path.join(IMG_PATH, IMG_FILE_EXTENSION))) 
    
    images = []
    smallest_size = (720, 720)
    
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
    pred = model.predict(test_img, batch_size=batch_size)
    
    cols = test.drop(['Index'], axis=1).columns
    # Set a threshold for converting probabilities to binary predictions
    threshold = 0.5

    # Apply the threshold to the predictions for binary labels
    binary_predictions = np.where(pred > threshold, 1, 0)

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
    combined_predictions_df['Aneurysm'] = combined_predictions_df.any(axis=1).astype(int)
    
    combined_predictions_df['Index'] = test['Index'].copy()
    
    logger.info("Saving output.csv...")
    combined_predictions_df.to_csv('./test_set/output.csv', index=False)
