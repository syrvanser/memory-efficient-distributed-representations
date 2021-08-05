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
from sklearn.preprocessing import normalize

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

import random

from models import MovielensModel, DPQMovielensModel, MGQEMovielensModel, NeuMFMovielensModel
from utils import NegativeSamplingDatasetWrapper, nDCG
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
parser.add_argument('--shared_centroids', type=bool, default=False)
parser.add_argument('--model', type=str, default="mf")
parser.add_argument('--download', type=bool, default=False)
parser.add_argument('--continue_from_checkpoint', type=int, default=0)
parser.add_argument('--partitions', type=int, default=2)


args = parser.parse_args()

random.seed(args.seed)
tf.random.set_seed(args.seed)
np.random.seed(args.seed)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'

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
validation = ratings[ratings['rank_latest'] == 2]
test = ratings[ratings['rank_latest'] == 1]

# drop columns that we no longer need
train = train[['user_id', 'movie_id', 'user_rating']]
test = test[['user_id', 'movie_id', 'user_rating']]
validation = validation[['user_id', 'movie_id', 'user_rating']]
train.loc[:, 'user_rating'] = 1
train_df = train


# distribution = train['movie_id'].value_counts().sort_index()
# p = np.asarray(distribution).astype('float64')
# p = p / np.sum(p)


#train = augment_data(train, unique_movie_ids, args.num_negatives)
train = NegativeSamplingDatasetWrapper(train, args, train['movie_id'].unique())
#validation = augment_data(validation, unique_movie_ids, args.num_negatives)

validation = pd.DataFrame.from_dict(validation).to_dict("list")
validation = {name: np.array(value, dtype=np.int32) for name, value in validation.items()}
# print(validation)
# print(len(validation['user_id']))
validation = tf.data.Dataset.from_tensor_slices(validation)

# eval_test = test.to_dict("list")
# eval_test = {name: np.array(value) for name, value in eval_test.items()}
# eval_test = tf.data.Dataset.from_tensor_slices(eval_test)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    if args.model == "mf":
        model = MovielensModel(args, unique_user_ids, unique_movie_ids)
    elif args.model == "dpq": 
        model = DPQMovielensModel(args, unique_user_ids, unique_movie_ids)
    elif args.model == "mgqe":
        user_freq = train_df[['user_id','movie_id']].groupby('user_id').agg('count')
        user_freq = user_freq.sort_values(by=['movie_id'])
        user_freq.reset_index(level=0, inplace=True)
        user_freq = user_freq.rename(columns={"user_id": "id", "movie_id": "freq"})
        movie_freq = train_df[['user_id','movie_id']].groupby('movie_id').agg('count')
        movie_freq = movie_freq.sort_values(by=['user_id'])
        movie_freq.reset_index(level=0, inplace=True)
        movie_freq = movie_freq.rename(columns={"movie_id": "id", "user_id": "freq"})

        model = MGQEMovielensModel(args, unique_user_ids, unique_movie_ids, user_freq, movie_freq)
    elif args.model == "mgqe_co":
        user_freq = train_df[['user_id','movie_id']].groupby('user_id').agg('count')
        train_pivoted = pd.pivot_table(train_df,values='user_rating',columns='movie_id',index='user_id').fillna(0)
        vals = []
        for index, row in train_pivoted.iterrows():
            ids = (row.where(row > 0).dropna().index.to_list()) 
            vals.append(sum(train_pivoted[ids].sum(axis=1))*row.sum())
        user_freq['movie_id'] = vals

        user_freq = user_freq.sort_values(by=['movie_id'])
        user_freq.reset_index(level=0, inplace=True)
        user_freq = user_freq.rename(columns={"user_id": "id", "movie_id": "freq"})


        movie_freq = train_df[['user_id','movie_id']].groupby('movie_id').agg('count')
        train_pivoted = pd.pivot_table(train_df,values='user_rating',columns='user_id',index='movie_id').fillna(0)
        vals = []
        for index, row in train_pivoted.iterrows():
            ids = (row.where(row > 0).dropna().index.to_list()) 
            vals.append(sum(train_pivoted[ids].sum(axis=1))*row.sum())
        
        movie_freq['user_id'] = vals

        movie_freq = movie_freq.sort_values(by=['user_id'])
        movie_freq.reset_index(level=0, inplace=True)
        movie_freq = movie_freq.rename(columns={"movie_id": "id", "user_id": "freq"})

        model = MGQEMovielensModel(args, unique_user_ids, unique_movie_ids, user_freq, movie_freq)

    elif args.model == "neumf":
        model = NeuMFMovielensModel(args, unique_user_ids, unique_movie_ids)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

    #model.ranking_model((["123"], ["42"]))

#cached_test = eval_test.batch(4096).cache() 
cached_validation = validation.batch(10000000).cache()

callbacks = []
callbacks.append(tf.keras.callbacks.EarlyStopping(
    monitor='val_total_loss', min_delta=0, patience=5, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
))

checkpoint_filepath = os.path.join(exp_dir, 'checkpoints')

callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_total_loss',
    mode='min',
    save_best_only=True)
)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# callbacks.append(TBCallback(logdir, histogram_freq=1))
# tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
if args.continue_from_checkpoint != 0:
    print("model loaded from checkpoint")
    model = tf.keras.models.load_model(checkpoint_filepath)
history = model.fit(train, epochs=args.epochs, verbose=1, validation_data=cached_validation, callbacks=callbacks)

epochs_list = [(x+1) for x in range(len(history.history["val_total_loss"]))]
plt.plot(epochs_list, history.history["val_total_loss"], label="validation loss")
plt.plot(epochs_list, history.history["total_loss"], label="training loss")
plt.title("Loss vs epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
# plt.show()


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
        
    predicted_labels = model((np.array([u]*100), np.array(test_items))).numpy().squeeze().tolist()
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