import tensorflow as tf
import tensorflow_recommenders as tfrs

from dpq_embedding import DPQEmbedding
from typing import Dict, Text


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
        self.ranking_model: tf.keras.Model = RankingModel(args, unique_user_ids, unique_movie_ids)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model(
            (features["user_id"], features["movie_id"]))
        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)

    def call(self, inputs):
        return self.ranking_model(inputs)


class DPQRankingModel(tf.keras.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids):
        super().__init__()

        # Compute embeddings for users.
        self.user_embeddings = DPQEmbedding(args.k, args.d, len(unique_user_ids) + 1, args.embedding_dimensions, activity_regularizer=tf.keras.regularizers.l2(args.l2_norm))

        # Compute embeddings for movies.
        self.movie_embeddings = DPQEmbedding(args.k, args.d, len(unique_movie_ids) + 1, args.embedding_dimensions, activity_regularizer=tf.keras.regularizers.l2(args.l2_norm))
    
        self.dot = tf.keras.layers.Dot(axes=1)

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
        return self.ratings(self.dot([user_embedding, movie_embedding]))

class DPQMovielensModel(tfrs.models.Model):

    def __init__(self, args, unique_user_ids, unique_movie_ids):
        super().__init__()
        self.ranking_model: tf.keras.Model = DPQRankingModel(args, unique_user_ids, unique_movie_ids)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
      )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model(
            (features["user_id"], features["movie_id"]))
        # The task computes the loss and the metrics.
        return self.task(labels=features["user_rating"], predictions=rating_predictions)

    def call(self, inputs):
        return self.ranking_model(inputs)