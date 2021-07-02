import tensorflow as tf
import tensorflow_recommenders as tfrs

from dpq_embedding import DPQEmbedding, MGQEEmbedding
from typing import Dict, Text
from utils import augment_data
class RankingModel(tf.keras.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids):
        super().__init__()

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.layers.Embedding(len(unique_user_ids) + 1, args.embedding_dimensions, embeddings_regularizer=tf.keras.regularizers.l2(args.l2_norm))

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, args.embedding_dimensions, embeddings_regularizer=tf.keras.regularizers.l2(args.l2_norm))
    
        # self.dot = tf.keras.layers.Dot(axes=1)

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(args.mlp_1,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.mlp_2,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.mlp_3,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="sigmoid")
        ])
    
    def call(self, inputs):
        user_id, movie_id = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_id)
        return self.ratings(tf.keras.layers.Multiply()([user_embedding, movie_embedding]))

class MovielensModel(tfrs.models.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids):
        super().__init__()
        self.args = args
        self.unique_user_ids = unique_user_ids
        self.unique_movie_ids = unique_movie_ids

        self.ranking_model: tf.keras.Model = RankingModel(args, unique_user_ids, unique_movie_ids)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        if type(features) == tuple:
            features = features[0]
        rating_predictions = self.ranking_model(
            (features["user_id"], features["movie_id"]))
        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)

    def call(self, inputs):
        if type(inputs) == tuple:
            return self.ranking_model(inputs)
        elif type(inputs) == dict:
            return  self.ranking_model((inputs["user_id"], inputs["movie_id"]))

class DPQRankingModel(tf.keras.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids):
        super().__init__()

        # Compute embeddings for users.
        self.user_embeddings = DPQEmbedding(args.k, args.d, len(unique_user_ids) + 1, args.embedding_dimensions, activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), share_subspace=args.share_subspace)

        # Compute embeddings for movies.
        self.movie_embeddings = DPQEmbedding(args.k, args.d, len(unique_movie_ids) + 1, args.embedding_dimensions, activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), share_subspace=args.share_subspace)
    
        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(args.mlp_1,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.mlp_2,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.mlp_3,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="sigmoid")
        ])
    
    def call(self, inputs):
        user_id, movie_id = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_id)
        return self.ratings(tf.keras.layers.Multiply()([user_embedding, movie_embedding]))

class DPQMovielensModel(tfrs.models.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids):
        super().__init__()
        self.ranking_model: tf.keras.Model = DPQRankingModel(args, unique_user_ids, unique_movie_ids)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
      )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        if type(features) == tuple:
            features = features[0]
        rating_predictions = self.ranking_model(
            (features["user_id"], features["movie_id"]))
        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)

    def call(self, inputs):
        if type(inputs) == tuple:
            return self.ranking_model(inputs)
        elif type(inputs) == dict:
            return  self.ranking_model((inputs["user_id"], inputs["movie_id"]))



class MGQERankingModel(tf.keras.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids, user_freqs, movie_freqs):
        super().__init__()

        # Compute embeddings for users.
        self.user_embeddings = MGQEEmbedding(args.k, args.d, len(unique_user_ids) + 1, args.embedding_dimensions, activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), frequencies=user_freqs, share_subspace=args.shared_centroids)

        # Compute embeddings for movies.
        self.movie_embeddings = MGQEEmbedding(args.k, args.d, len(unique_movie_ids) + 1, args.embedding_dimensions, activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), frequencies=movie_freqs, share_subspace=args.shared_centroids)
    
        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(args.mlp_1,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.mlp_2,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.mlp_3,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="sigmoid")
        ])
    
    def call(self, inputs):
        user_id, movie_id = inputs
        

        user_embedding = self.user_embeddings(tf.cast(user_id, tf.int64))
        movie_embedding = self.movie_embeddings(tf.cast(movie_id, tf.int64))
        return self.ratings(tf.keras.layers.Multiply()([user_embedding, movie_embedding]))

class MGQEMovielensModel(tfrs.models.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids, user_freqs, movie_freqs):
        super().__init__()
        self.ranking_model: tf.keras.Model = MGQERankingModel(args, unique_user_ids, unique_movie_ids, user_freqs, movie_freqs)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
      )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        if type(features) == tuple:
            features = features[0]
        rating_predictions = self.ranking_model(
            (features["user_id"], features["movie_id"]))
        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)

    def call(self, inputs):
        if type(inputs) == tuple:
            return self.ranking_model(inputs)
        elif type(inputs) == dict:
            return  self.ranking_model((inputs["user_id"], inputs["movie_id"]))

class NeuMFRankingModel(tf.keras.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids):
        super().__init__()

        # Compute embeddings for users.
        self.user_embeddings1 = tf.keras.layers.Embedding(len(unique_user_ids) + 1, args.embedding_dimensions, embeddings_regularizer=tf.keras.regularizers.l2(args.l2_norm))
        self.user_embeddings2 = tf.keras.layers.Embedding(len(unique_user_ids) + 1, args.embedding_dimensions, embeddings_regularizer=tf.keras.regularizers.l2(args.l2_norm))

        # Compute embeddings for movies.
        self.movie_embeddings1 = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, args.embedding_dimensions, embeddings_regularizer=tf.keras.regularizers.l2(args.l2_norm))
        self.movie_embeddings2 = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, args.embedding_dimensions, embeddings_regularizer=tf.keras.regularizers.l2(args.l2_norm))
    

        # self.dot = tf.keras.layers.Dot(axes=1)

        # Compute predictions.
        self.mlp = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(args.mlp_1,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.mlp_2,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.mlp_3,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            # Make rating predictions in the final layer.
        ])

        self.concat = tf.keras.layers.Concatenate(axis=1)

        self.out = tf.keras.layers.Dense(1,  activity_regularizer=tf.keras.regularizers.l2(args.l2_norm), activation="sigmoid")
    
    def call(self, inputs):
        user_id, movie_id = inputs

        user_embedding1 = self.user_embeddings1(user_id)
        movie_embedding1 = self.movie_embeddings1(movie_id)

        user_embedding2 = self.user_embeddings2(user_id)
        movie_embedding2 = self.movie_embeddings2(movie_id)

        prod = tf.keras.layers.Multiply()([user_embedding1, movie_embedding1])
        mlp = self.mlp(self.concat([user_embedding2, movie_embedding2]))
        return self.out(self.concat([prod, mlp]))

class NeuMFMovielensModel(tfrs.models.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids):
        super().__init__()
        self.ranking_model: tf.keras.Model = NeuMFRankingModel(args, unique_user_ids, unique_movie_ids)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        if type(features) == tuple:
            features = features[0]
        rating_predictions = self.ranking_model(
            (features["user_id"], features["movie_id"]))
        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)

    def call(self, inputs):
        if type(inputs) == tuple:
            return self.ranking_model(inputs)
        elif type(inputs) == dict:
            return  self.ranking_model((inputs["user_id"], inputs["movie_id"]))