#!/usr/bin/env python

import zipfile
import tensorflow as tf
import requests
import os
from tqdm import tqdm

import src

def load_and_unzip_data(src_link, raw_path, clean_path, filename=None):
    cwd = os.getcwd()
    raw_cwd = f"{os.getcwd().rsplit('/', 1)[0]}/{raw_path}"
    clean_cwd = f"{os.getcwd().rsplit('/', 1)[0]}/{clean_path}"
    os.makedirs(raw_cwd, exist_ok=True)
    os.makedirs(clean_cwd, exist_ok=True)
    

    if not filename:
      filename = src_link.split('/')[-1]
    
    os.chdir(raw_cwd)
    # Streaming, so we can iterate over the response.
    response = requests.get(src_link, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as zfile:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            zfile.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    
    zip_ref = zipfile.ZipFile(filename, "r")
    os.chdir(clean_cwd)
    zip_ref.extractall()
    zip_ref.close()
    os.chdir(cwd)

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img
