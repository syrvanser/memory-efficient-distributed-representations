import math
import numpy as np
import pandas as pd

from tqdm import tqdm

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


def augment_data(data, unique_movie_ids, num_negatives=4):
    users, items, labels = [], [], []
    # This is the set of items that each user has interaction with
    user_item_set = set(zip(data['user_id'], data['movie_id']))

    # 4:1 ratio of negative to positive samples
    
    for (u, i) in tqdm(user_item_set):
        users.append(u)
        items.append(i)
        labels.append(1) # items that the user has interacted with are positive
        for _ in range(num_negatives):
            # randomly select an item
            negative_item = np.random.choice(unique_movie_ids) 
            # check that the user has not interacted with this item
            while (u, negative_item) in user_item_set:
                negative_item = np.random.choice(unique_movie_ids)
            users.append(u)
            items.append(negative_item)
            labels.append(0) # items not interacted with are negative
    return pd.DataFrame({'user_id': users, 'movie_id': items, 'user_rating': labels})
