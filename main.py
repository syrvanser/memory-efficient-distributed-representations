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

from models import MovielensModel, DPQMovielensModel
from utils import nDCG, augment_data
from callback import TBCallback

# %matplotlib inline
# %load_ext tensorboard

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--l2_norm', type=float, default=0.00002)
parser.add_argument('--embedding_dimensions', type=int, default=8)
parser.add_argument('--num_negatives', type=int, default=4)
parser.add_argument('--mlp_1', type=int, default=16)
parser.add_argument('--mlp_2', type=int, default=8)
parser.add_argument('--mlp_3', type=int, default=4)
parser.add_argument('--dataset', type=str, default="100k")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--exp_dir', type=str, default="temp")
parser.add_argument('--k', type=int, default=32)
parser.add_argument('--d', type=int, default=8)
parser.add_argument('--shared_centroids', type=bool, default=True)
parser.add_argument('--model', type=str, default="mf")
parser.add_argument('--download', type=bool, default=False)


args = parser.parse_args()

random.seed(args.seed)
tf.random.set_seed(args.seed)
np.random.seed(args.seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

if args.dataset == "100k":
    dataset = "movielens/100k-ratings"
    args.ds_size = 100000
elif args.dataset == "1m":
    dataset = "movielens/1m-ratings"
    args.ds_size = 1000000
elif args.dataset == "20m":
    dataset = "movielens/20m-ratings"
    args.ds_size = 20000000
else:
    raise ValueError("Unknown dataset")

# NOTE: setting up logging and saving
dir_path = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(dir_path, 'results', args.exp_dir)
os.makedirs(exp_dir, exist_ok=True)

# now make variables for all the relevant files we'll make
log_path = os.path.join(exp_dir, 'log.log')
#logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f'Running experiment with arguments {args}')

ratings = tfds.load(dataset, data_dir=os.path.join(dir_path, 'data'), split="train", download=args.download, shuffle_files=True)
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

unique_movie_ids = ratings['movie_id'].unique()
unique_user_ids = ratings['user_id'].unique()

train = ratings[ratings['rank_latest'] > 2]
validation = ratings[ratings['rank_latest'] == 2]
test = ratings[ratings['rank_latest'] == 1]

# drop columns that we no longer need
train = train[['user_id', 'movie_id', 'user_rating']]
test = test[['user_id', 'movie_id', 'user_rating']]
validation = validation[['user_id', 'movie_id', 'user_rating']]
train.loc[:, 'user_rating'] = 1


train = augment_data(train, unique_movie_ids, args.num_negatives)
train = train.to_dict("list")
train = {name: np.array(value) for name, value in train.items()}
train = tf.data.Dataset.from_tensor_slices(train)

validation = augment_data(validation, unique_movie_ids, args.num_negatives)

validation = validation.to_dict("list")
validation = {name: np.array(value) for name, value in validation.items()}
validation = tf.data.Dataset.from_tensor_slices(validation)

eval_test = test.to_dict("list")
eval_test = {name: np.array(value) for name, value in eval_test.items()}
eval_test = tf.data.Dataset.from_tensor_slices(eval_test)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    if args.model == "mf":
        model = MovielensModel(args, unique_user_ids, unique_movie_ids)
    elif args.model == "dpq": 
        model = DPQMovielensModel(args, unique_user_ids, unique_movie_ids)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
    model.ranking_model((["123"], ["42"]))

cached_train = train.shuffle(args.ds_size, seed=args.seed).batch(args.batch_size).cache()
cached_test = eval_test.batch(4096).cache()
cached_validation = validation.batch(4096).cache()

callbacks = []
# callbacks.append(tf.keras.callbacks.EarlyStopping(
#     monitor='val_total_loss', min_delta=0, patience=10, verbose=0,
#     mode='auto', baseline=None, restore_best_weights=True
# ))

checkpoint_filepath = os.path.join(exp_dir, 'checkpoints')

callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_total_loss',
    mode='min',
    save_best_only=True)
)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# callbacks.append(TBCallback(logdir, histogram_freq=1))

history = model.fit(cached_train, epochs=args.epochs, verbose=1, validation_data=cached_validation, callbacks=callbacks)

epochs_list = [(x+1) for x in range(len(history.history["val_total_loss"]))]
plt.plot(epochs_list, history.history["val_total_loss"], label="validation loss")
plt.plot(epochs_list, history.history["total_loss"], label="training loss")
plt.title("Loss vs epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
# plt.show()


logging.info(model.evaluate(cached_test, return_dict=True))

test_user_item_set = set(zip(test['user_id'], test['movie_id']))

user_interacted_items = ratings.groupby('user_id')['movie_id'].apply(list).to_dict()

hits = []
ndcg_results = []

for (u,i) in tqdm(test_user_item_set):
    interacted_items = user_interacted_items[u]
    not_interacted_items = set(unique_movie_ids) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + [i]
    random.shuffle(test_items)
        
    predicted_labels = model.ranking_model(([u]*100, test_items)).numpy().squeeze().tolist()
    top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
    top10_filtered = [int(item == i) for item in top10_items]
    
    if i in top10_items:
        hits.append(1)
    else:
        hits.append(0)

    ideal = [1] + [0] * 99
    ndcg_results.append(nDCG(top10_filtered, 10, ideal))

logging.info("RESULTS:")    
logging.info("Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))
logging.info("nDCG @ 10 is {:.2f}".format(np.average(ndcg_results)))
