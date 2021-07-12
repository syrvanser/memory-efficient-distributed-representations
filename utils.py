import math
import numpy as np
import tensorflow as tf
import random

def nDCG(docs, n, ideal_docs):
    gain_sum = docs[0] 
    for i in range(2, min(len(docs)+1, n+1)):
        gain_sum += docs[i-1]/math.log(i,2)
    ideal_sum = ideal_docs[0]
    for i in range(2, min(n+1, len(ideal_docs)+1)):
        ideal_sum += ideal_docs[i-1]/math.log(i,2)
    return gain_sum/ideal_sum

def hit_ratio_at_n(docs, n):
    for i in range(min(n, len(docs))):
        if docs[i] > 0:
            return 1
    return 0

def augment_data(data, unique_movie_ids, num_negatives=4, distribution = None):
    users, items, labels = [], [], []
    # This is the set of items that each user has interaction with
    user_item_set = set(zip(data['user_id'], data['movie_id']))
    # 4:1 ratio of negative to positive samples
    for (u, i) in user_item_set:
        users.append(u)
        items.append(i)
        labels.append(1) # items that the user has interacted with are positive
        for _ in range(num_negatives):
            # randomly select an item
            negative_item = np.random.choice(unique_movie_ids, p=distribution) 
            # check that the user has not interacted with this item
            while (u, negative_item) in user_item_set:
                negative_item = np.random.choice(unique_movie_ids, p=distribution)
            users.append(u)
            items.append(negative_item)
            labels.append(0) # items not interacted with are negative
    # tf.print(tf.convert_to_tensor(users, dtype=tf.int64))
    shuffled = list(zip(users, items, labels))
    #print(shuffled[0:5])
    random.shuffle(shuffled)
    users, items, labels = zip(*shuffled)
    #tf.print({'user_id': tf.concat([data['user_id'], tf.convert_to_tensor(users, dtype=tf.int64)], axis=0), 'movie_id':  tf.concat([data['user_id'], tf.convert_to_tensor(items, dtype=tf.int64)], axis=0), 'user_rating':  tf.concat([data['user_id'], tf.convert_to_tensor(labels, dtype=tf.int64)], axis=0)})
    # return {'user_id': tf.concat([data['user_id'], tf.convert_to_tensor(users, dtype=tf.int64)], axis=0), 'movie_id':  tf.concat([data['user_id'], tf.convert_to_tensor(items, dtype=tf.int64)], axis=0), 'user_rating':  tf.concat([data['user_id'], tf.convert_to_tensor(labels, dtype=tf.int64)], axis=0)}
    return {'user_id': users, 'movie_id':  items, 'user_rating': labels}

class NegativeSamplingDatasetWrapper(tf.keras.utils.Sequence):
    def __init__(self, data, args, unique_movie_ids, distribution=None):
        self.positives = data
        self.distribution = distribution
        self.current = augment_data(self.positives, unique_movie_ids, args.num_negatives, distribution)
        self.unique_movie_ids = unique_movie_ids
        self.args = args
        # train = data.to_dict("list")
        # train = {name: np.array(value) for name, value in train.items()}
        # train = tf.data.Dataset.from_tensor_slices(train)
        
        # self.cached_train = train.shuffle(args.ds_size * (args.num_negatives + 1), seed=args.seed).batch(args.batch_size).cache()
        self.batch_size = args.batch_size


    def __len__(self):
        return math.ceil(len(self.current['user_id']) / self.batch_size)

    def on_epoch_end(self):
        #print('on_epoch_end')
        self.current = augment_data(self.positives, self.unique_movie_ids, self.args.num_negatives, self.distribution)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch =  {'user_id': tf.convert_to_tensor(self.current['user_id'][start:end], dtype=tf.int32), 
                'movie_id':  tf.convert_to_tensor(self.current['movie_id'][start:end], dtype=tf.int32), 
                'user_rating': tf.convert_to_tensor(self.current['user_rating'][start:end], dtype=tf.int32)}
        return batch 

