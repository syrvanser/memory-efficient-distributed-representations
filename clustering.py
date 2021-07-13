import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import sys
import datetime
import pandas as pd
pd.set_option('display.max_colwidth', None)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from tqdm import tqdm

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

import random

from models import MovielensModel, DPQMovielensModel, MGQEMovielensModel, NeuMFMovielensModel
from utils import NegativeSamplingDatasetWrapper, nDCG, augment_data
from callback import TBCallback

# %matplotlib inline
# %load_ext tensorboard
class argsCls:
    pass
args = argsCls()
args.seed =0
args.num_negatives = 7
args.batch_size = 256

random.seed(args.seed)
tf.random.set_seed(args.seed)
np.random.seed(args.seed)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'

dataset = "movielens/1m-ratings"
args.ds_size = 1000000

# NOTE: setting up logging and saving
dir_path = os.path.dirname(os.path.realpath(__file__))

#logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f'Running experiment with arguments {args}')

ratings = tfds.load(dataset, data_dir=os.path.join(dir_path, 'data'), split="train", shuffle_files=True)
ratings = ratings.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
    "timestamp": x["timestamp"]
})

# shuffled = ratings.shuffle(args.ds_size, reshuffle_each_iteration=False, seed=args.seed)
ratings = tfds.as_dataframe(ratings.take(-1))

ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'] \
                                .rank(method='first', ascending=False)
ratings[['user_id', 'movie_id']] = ratings[['user_id', 'movie_id']].apply(pd.to_numeric)

unique_movie_ids = ratings['movie_id'].unique()
unique_user_ids = ratings['user_id'].unique()

user2user_encoded = {x: i for i, x in enumerate(unique_user_ids.tolist())}
user_encoded2user = {i: x for i, x in enumerate(unique_user_ids.tolist())}
movie2movie_encoded = {x: i for i, x in enumerate(unique_movie_ids.tolist())}
movie_encoded2movie = {i: x for i, x in enumerate(unique_movie_ids.tolist())}

ratings['user_id'] = ratings['user_id'].map(user2user_encoded)
ratings['movie_id'] = ratings['movie_id'].map(movie2movie_encoded)

unique_movie_ids = ratings['movie_id'].unique()
unique_user_ids = ratings['user_id'].unique()

train = ratings[ratings['rank_latest'] > 2]
# validation = ratings[ratings['rank_latest'] == 2]
# test = ratings[ratings['rank_latest'] == 1]

# drop columns that we no longer need
train = train[['user_id', 'movie_id', 'user_rating']]
# test = test[['user_id', 'movie_id', 'user_rating']]
# validation = validation[['user_id', 'movie_id', 'user_rating']]
train.loc[:, 'user_rating'] = 1

#train = augment_data(train, unique_movie_ids, args.num_negatives)
#!!!train = NegativeSamplingDatasetWrapper(train, args, unique_movie_ids)
#validation = augment_data(validation, unique_movie_ids, args.num_negatives)

# validation = pd.DataFrame.from_dict(validation).to_dict("list")
# validation = {name: np.array(value, dtype=np.int32) for name, value in validation.items()}
# # print(validation)
# # print(len(validation['user_id']))
# validation = tf.data.Dataset.from_tensor_slices(validation)

print(train.head(50))