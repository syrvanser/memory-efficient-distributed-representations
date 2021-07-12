from re import I
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import defaultdict

class DPQEmbedding(tf.keras.layers.Layer):

    def __init__(self, k, d, vocab_size, embebdding_size, share_subspace=False, **kwargs):
        super(DPQEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embebdding_size
        self.share_subspace = share_subspace
        self.k = k
        self.d = d
        self.sub_size = embebdding_size//d
        self.query_wemb = self.add_weight(name='emb_table', shape=[self.vocab_size, self.embedding_size], dtype=tf.float32, initializer="he_normal", trainable=True)
        #print("emb shape:", self.vocab_size, self.embedding_size)
        self.kdq = KDQuantizer(self.k, self.d, self.sub_size, self.share_subspace)

    def call(self, inputs, training=None): # was D * subs_size before TODO
        inputs = tf.cast(inputs, tf.int32)
        idxs = tf.reshape(inputs, [-1]) #flatten 

        input_emb = tf.nn.embedding_lookup(self.query_wemb, idxs) 

        _, input_emb = self.kdq(tf.reshape(input_emb, [-1, self.d, self.sub_size]), training=training)
        final_size = tf.concat([tf.shape(inputs), tf.constant([self.embedding_size])], 0)
        input_emb = tf.reshape(input_emb, final_size)
        return input_emb

class KDQuantizer(tf.keras.layers.Layer):
    def __init__(self, k, d, sub_size, shared_centroids=False, beta=0., tau=1.0, **kwargs):
        """
            Args:
            K, D: int, size of KD code.
            d_in: dim of continuous input for each of D axis.
            d_out: dim of continuous output for each of D axis.
            tie_in_n_out: boolean, whether or not to tie the input/output centroids.
                If True, it is vector quantization, else it is tempering softmax.
            query_metric: string, which metric to use for input centroid matching.
            shared_centroids: boolean, whether or not to share centroids for
                different bits in D.
            beta: float, KDQ regularization coefficient.
            tau: float or None, (tempering) softmax temperature.
                If None, set to learnable.
            softmax_BN: whether to use BN in (tempering) softmax.
        """
        super(KDQuantizer, self).__init__(**kwargs)
        self.k = k
        self.d = d
        self.sub_size = sub_size
        self.shared_centroids = shared_centroids
        self.beta = beta
        self.tau = tf.constant(tau)

        # Create centroids for keys and values.
        d_to_create = 1 if shared_centroids else d
        self.centroids_k = self.add_weight(name='centroids', shape=[d_to_create, k, sub_size], initializer="random_normal", trainable = True)
        #print("centroids_shape:", d_to_create, k, sub_size)

        if shared_centroids:
            self.centroids_k = tf.tile(self.centroids_k, [d, 1, 1])

        self.batch_norm = tf.keras.layers.BatchNormalization(scale=False, center=False)

    def call(self, inputs, training=True):
        """Returns quantized embeddings from centroids.
        Args:
        inputs: embedding tensor of shape (batch_size, D, d_in)
        Returns:
        code: (batch_size, D)
        embs_quantized: (batch_size, D, d_out)
        """
        # Compute distance (in a metric) between inputs and centroids_k
        # the response is in the shape of (batch_size, D, K)
        norm_1 = tf.math.reduce_sum(inputs**2, -1, keepdims=True)  # (bs, D, 1)
        norm_2 = tf.expand_dims(tf.reduce_sum(self.centroids_k**2, -1), 0)  # (1, D, K)
        dot = tf.matmul(tf.transpose(inputs, perm=[1, 0, 2]),
                        tf.transpose(self.centroids_k, perm=[0, 2, 1]))  # (D, bs, K)
        response = -norm_1 + 2 * tf.transpose(dot, perm=[1, 0, 2]) - norm_2
        response = tf.reshape(response, [-1, self.d, self.k])
        response = self.batch_norm(response, training=training)

        # Compute the codes based on response.
        codes = tf.argmax(response, -1, output_type=tf.int32)  # (batch_size, D)
        neighbor_idxs = codes

        # Compute the outputs, which has shape (batch_size, D, d_out)
        if not self.shared_centroids:
            D_base = tf.convert_to_tensor(
                [self.k * d for d in range(self.d)], dtype=tf.int32)
            neighbor_idxs += tf.expand_dims(D_base, 0)  # (batch_size, D)
        neighbor_idxs = tf.reshape(neighbor_idxs, [-1])  # (batch_size * D)
        centroids_k = tf.reshape(self.centroids_k, [-1, self.sub_size])
        outputs = tf.nn.embedding_lookup(centroids_k, neighbor_idxs)
        outputs = tf.reshape(outputs, [-1, self.d, self.sub_size])
        outputs_final = tf.stop_gradient(outputs - inputs) + inputs
        
        # Add regularization for updating centroids / stabilization.
        if training:
            alpha = 1.
            beta = self.beta
            gamma = 0.0
            reg = alpha * tf.reduce_mean(
                (outputs - tf.stop_gradient(inputs))**2, name="centroids_adjust")
            reg += beta * tf.reduce_mean(
                (tf.stop_gradient(outputs) - inputs)**2, name="input_commit")
            minaxis = [0, 1] if self.shared_centroids else [0]
            reg += gamma * tf.reduce_mean(  # could sg(inputs), but still not eff.
                tf.reduce_min(-response, minaxis), name="de_isolation")
    
            self.add_loss(reg)

        return codes, outputs_final

class MGQEEmbedding(tf.keras.layers.Layer):

    def __init__(self, k, d, vocab_size, embebdding_size, frequencies, share_subspace=False, **kwargs):
        super(MGQEEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embebdding_size
        self.share_subspace = share_subspace
        n = 80
        cutoff = int(len(frequencies)*(n/100))
        tail = frequencies[:cutoff]
        head = frequencies[cutoff:]
        
        self.dictionary = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant( head['id'].tolist() + tail['id'].tolist()),
            values=tf.constant([0] * len(head) + [1] * len(tail)),
        ),
        default_value=tf.constant(k),
        name="partitions"
        )

        #print(head.head())
        

        self.k = k
        self.d = d
        self.sub_size = embebdding_size//d
        self.query_wemb = self.add_weight(name='emb_table', shape=[self.vocab_size, self.embedding_size], dtype=tf.float32, initializer="he_normal", trainable=True)
        self.kdq = PartialKDQuantizer(self.k, self.d, self.sub_size, self.dictionary, self.share_subspace)

    def call(self, inputs, training=None): # was D * subs_size before TODO
        inputs = tf.cast(inputs, tf.int32)
        #print("shape: ", inputs.shape)
        idxs = tf.reshape(inputs, [-1]) #flatten 
        #print(idxs)
        partitions = tf.map_fn(fn=lambda t: self.dictionary[t], elems=idxs)
        #print(partitions)
        input_emb = tf.nn.embedding_lookup(self.query_wemb, idxs) 
        _, input_emb = self.kdq(input_emb, partitions=partitions, training=training)
        final_size = tf.concat([tf.shape(inputs), tf.constant([self.embedding_size])], 0)
        input_emb = tf.reshape(input_emb, final_size)
        return input_emb

class PartialKDQuantizer(tf.keras.layers.Layer):
    def __init__(self, k, d, sub_size, dictionary, shared_centroids=False, beta=0., tau=1.0, **kwargs):
        """
            Args:
            K, D: int, size of KD code.
            d_in: dim of continuous input for each of D axis.
            d_out: dim of continuous output for each of D axis.
            tie_in_n_out: boolean, whether or not to tie the input/output centroids.
                If True, it is vector quantization, else it is tempering softmax.
            query_metric: string, which metric to use for input centroid matching.
            shared_centroids: boolean, whether or not to share centroids for
                different bits in D.
            beta: float, KDQ regularization coefficient.
            tau: float or None, (tempering) softmax temperature.
                If None, set to learnable.
            softmax_BN: whether to use BN in (tempering) softmax.
        """
        super(PartialKDQuantizer, self).__init__(**kwargs)
        self.k = k
        self.d = d
        self.sub_size = sub_size
        self.shared_centroids = shared_centroids
        self.beta = beta
        self.tau = tf.constant(tau)
        self.dictionary = dictionary

        # Create centroids for keys and values.
        d_to_create = 1 if shared_centroids else d
        self.centroids_k = self.add_weight(name='centroids', shape=[d_to_create, k, sub_size], initializer="random_normal", trainable = True)
        
        if shared_centroids:
            self.centroids_k = tf.tile(self.centroids_k, [d, 1, 1])

        self.batch_norm1 = tf.keras.layers.BatchNormalization(scale=False, center=False)
        self.batch_norm2 = tf.keras.layers.BatchNormalization(scale=False, center=False)

    def call(self, inputs, partitions, training=True):
        """Returns quantized embeddings from centroids.
        Args:
        inputs: embedding tensor of shape (batch_size, D, d_in)  256 * dims(32)
        Returns:
        code: (batch_size, D)
        embs_quantized: (batch_size, D, d_out)
        """
        # Compute distance (in a metric) between inputs and centroids_k
        # the response is in the shape of (batch_size, D, K)
        #available_centroids = tf.slice(self.centroids_k, [0, 0, 0], [0, self.k, 0])
        #print(inputs.shape)
        
 
        # inputs_head_ids = tf.where(tf.equal(partitions, self.k))
        # inputs_tail_ids = tf.where(tf.equal(partitions, self.k//4))
 
        # head = tf.gather(inputs, inputs_head_ids)
        # tail = tf.gather(inputs, inputs_tail_ids)

        head, tail = tf.dynamic_partition(inputs, partitions=partitions, num_partitions=2)
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(inputs)[0]), partitions=partitions, num_partitions=2)

        inputs = tf.reshape(inputs, [-1, self.d, self.sub_size])
        # print(available_centroids.shape)
        available_centroids = tf.slice(self.centroids_k, [0, 0, 0], [-1, self.k, -1])
        head = tf.reshape(head, [-1, self.d, self.sub_size])

        norm_1 = tf.math.reduce_sum(head**2, -1, keepdims=True)  # (bs, D, 1)
        norm_2 = tf.expand_dims(tf.reduce_sum(available_centroids**2, -1), 0)  # (1, D, K)
        dot = tf.matmul(tf.transpose(head, perm=[1, 0, 2]),
                        tf.transpose(available_centroids, perm=[0, 2, 1]))  # (D, bs, K)
        response = -norm_1 + 2 * tf.transpose(dot, perm=[1, 0, 2]) - norm_2
        response = tf.reshape(response, [-1, self.d, self.k])
        response = self.batch_norm1(response, training=training)
        codes_head = tf.argmax(response, -1, output_type=tf.int32)

        available_centroids = tf.slice(self.centroids_k, [0, 0, 0], [-1, self.k//4, -1])

        tail = tf.reshape(tail, [-1, self.d, self.sub_size])
        norm_1 = tf.math.reduce_sum(tail**2, -1, keepdims=True)  # (bs, D, 1)
        norm_2 = tf.expand_dims(tf.reduce_sum(available_centroids**2, -1), 0)  # (1, D, K)
        dot = tf.matmul(tf.transpose(tail, perm=[1, 0, 2]),
                        tf.transpose(available_centroids, perm=[0, 2, 1]))  # (D, bs, K)
        response = -norm_1 + 2 * tf.transpose(dot, perm=[1, 0, 2]) - norm_2
        response = tf.reshape(response, [-1, self.d, self.k//4])
        response = self.batch_norm2(response, training=training)
        codes_tails = tf.argmax(response, -1, output_type=tf.int32)

        # combined = tf.concat([codes_head, codes_tails], 0)
        # order = tf.reshape(tf.argsort(tf.concat([inputs_head_ids, inputs_tail_ids], 0), 0), [-1])
        # codes = tf.gather(combined, order)
        codes = tf.dynamic_stitch(condition_indices, [codes_head, codes_tails])

        # Compute the codes based on response.

        neighbor_idxs = codes

        # Compute the outputs, which has shape (batch_size, D, d_out)
        if not self.shared_centroids:
            D_base = tf.convert_to_tensor(
                [self.k * d for d in range(self.d)], dtype=tf.int32)
            neighbor_idxs += tf.expand_dims(D_base, 0)  # (batch_size, D)
        neighbor_idxs = tf.reshape(neighbor_idxs, [-1])  # (batch_size * D)
        centroids_k = tf.reshape(self.centroids_k, [-1, self.sub_size])
        
        outputs = tf.nn.embedding_lookup(centroids_k, neighbor_idxs)
        outputs = tf.reshape(outputs, [-1, self.d, self.sub_size])
        outputs_final = tf.stop_gradient(outputs - inputs) + inputs
        
        # Add regularization for updating centroids / stabilization.
        if training:
            alpha = 1.
            beta = self.beta
            gamma = 0.0
            reg = alpha * tf.reduce_mean(
                (outputs - tf.stop_gradient(inputs))**2, name="centroids_adjust")
            reg += beta * tf.reduce_mean(
                (tf.stop_gradient(outputs) - inputs)**2, name="input_commit")
            minaxis = [0, 1] if self.shared_centroids else [0]
            reg += gamma * tf.reduce_mean(  # could sg(inputs), but still not eff.
                tf.reduce_min(-response, minaxis), name="de_isolation")
        
            self.add_loss(reg)
        

        return codes, outputs_final

class TripleMGQEEmbedding(tf.keras.layers.Layer):

    def __init__(self, k, d, vocab_size, embebdding_size, frequencies, share_subspace=False, **kwargs):
        super(TripleMGQEEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embebdding_size
        self.share_subspace = share_subspace
        n_1 = 99
        n_2 =90
        cutoff_1 = int(len(frequencies)*(n_1/100))
        cutoff_2 = int(len(frequencies)*(n_2/100))
        tail = frequencies[:cutoff_1]
        mid = frequencies[cutoff_1:cutoff_2]
        head = frequencies[cutoff_2:]
        
        self.dictionary = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant( head['id'].tolist() +  mid['id'].tolist() + tail['id'].tolist()),
            values=tf.constant([0] * len(head) + [1] * len(mid) + [2] * len(tail)),
        ),
        default_value=tf.constant(k),
        name="partitions"
        )

        #print(head.head())
        

        self.k = k
        self.d = d
        self.sub_size = embebdding_size//d
        self.query_wemb = self.add_weight(name='emb_table', shape=[self.vocab_size, self.embedding_size], dtype=tf.float32, initializer="he_normal", trainable=True)
        self.kdq = TriplePartialKDQuantizer(self.k, self.d, self.sub_size, self.dictionary, self.share_subspace)

    def call(self, inputs, training=None): # was D * subs_size before TODO
        inputs = tf.cast(inputs, tf.int32)
        #print("shape: ", inputs.shape)
        idxs = tf.reshape(inputs, [-1]) #flatten 
        #print(idxs)
        partitions = tf.map_fn(fn=lambda t: self.dictionary[t], elems=idxs)
        #print(partitions)
        input_emb = tf.nn.embedding_lookup(self.query_wemb, idxs) 
        _, input_emb = self.kdq(input_emb, partitions=partitions, training=training)
        final_size = tf.concat([tf.shape(inputs), tf.constant([self.embedding_size])], 0)
        input_emb = tf.reshape(input_emb, final_size)
        return input_emb

class TriplePartialKDQuantizer(tf.keras.layers.Layer):
    def __init__(self, k, d, sub_size, dictionary, shared_centroids=False, beta=0., tau=1.0, **kwargs):
        """
            Args:
            K, D: int, size of KD code.
            d_in: dim of continuous input for each of D axis.
            d_out: dim of continuous output for each of D axis.
            tie_in_n_out: boolean, whether or not to tie the input/output centroids.
                If True, it is vector quantization, else it is tempering softmax.
            query_metric: string, which metric to use for input centroid matching.
            shared_centroids: boolean, whether or not to share centroids for
                different bits in D.
            beta: float, KDQ regularization coefficient.
            tau: float or None, (tempering) softmax temperature.
                If None, set to learnable.
            softmax_BN: whether to use BN in (tempering) softmax.
        """
        super(TriplePartialKDQuantizer, self).__init__(**kwargs)
        self.k = k
        self.d = d
        self.sub_size = sub_size
        self.shared_centroids = shared_centroids
        self.beta = beta
        self.tau = tf.constant(tau)
        self.dictionary = dictionary

        # Create centroids for keys and values.
        d_to_create = 1 if shared_centroids else d
        self.centroids_k = self.add_weight(name='centroids', shape=[d_to_create, k, sub_size], initializer="random_normal", trainable = True)
        
        if shared_centroids:
            self.centroids_k = tf.tile(self.centroids_k, [d, 1, 1])

        self.batch_norm1 = tf.keras.layers.BatchNormalization(scale=False, center=False)
        self.batch_norm2 = tf.keras.layers.BatchNormalization(scale=False, center=False)
        self.batch_norm3 = tf.keras.layers.BatchNormalization(scale=False, center=False)

    def call(self, inputs, partitions, training=True):
        """Returns quantized embeddings from centroids.
        Args:
        inputs: embedding tensor of shape (batch_size, D, d_in)  256 * dims(32)
        Returns:
        code: (batch_size, D)
        embs_quantized: (batch_size, D, d_out)
        """
        # Compute distance (in a metric) between inputs and centroids_k
        # the response is in the shape of (batch_size, D, K)
        #available_centroids = tf.slice(self.centroids_k, [0, 0, 0], [0, self.k, 0])
        #print(inputs.shape)
        
 
        # inputs_head_ids = tf.where(tf.equal(partitions, self.k))
        # inputs_mid_ids = tf.where(tf.equal(partitions, self.k//2))
        # inputs_tail_ids = tf.where(tf.equal(partitions, self.k//8))
 
        # head = tf.gather(inputs, inputs_head_ids)
        # mid = tf.gather(inputs, inputs_mid_ids)
        # tail = tf.gather(inputs, inputs_tail_ids)

        head, mid, tail = tf.dynamic_partition(inputs, partitions=partitions, num_partitions=3)
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(inputs)[0]), partitions=partitions, num_partitions=3)

        inputs = tf.reshape(inputs, [-1, self.d, self.sub_size])
        # print(available_centroids.shape)
        available_centroids = tf.slice(self.centroids_k, [0, 0, 0], [-1, self.k, -1])
        head = tf.reshape(head, [-1, self.d, self.sub_size])

        norm_1 = tf.math.reduce_sum(head**2, -1, keepdims=True)  # (bs, D, 1)
        norm_2 = tf.expand_dims(tf.reduce_sum(available_centroids**2, -1), 0)  # (1, D, K)
        dot = tf.matmul(tf.transpose(head, perm=[1, 0, 2]),
                        tf.transpose(available_centroids, perm=[0, 2, 1]))  # (D, bs, K)
        response = -norm_1 + 2 * tf.transpose(dot, perm=[1, 0, 2]) - norm_2
        response = tf.reshape(response, [-1, self.d, self.k])
        response = self.batch_norm1(response, training=training)
        codes_head = tf.argmax(response, -1, output_type=tf.int32)

        available_centroids = tf.slice(self.centroids_k, [0, 0, 0], [-1, self.k//2, -1])

        mid = tf.reshape(mid, [-1, self.d, self.sub_size])
        norm_1 = tf.math.reduce_sum(mid**2, -1, keepdims=True)  # (bs, D, 1)
        norm_2 = tf.expand_dims(tf.reduce_sum(available_centroids**2, -1), 0)  # (1, D, K)
        dot = tf.matmul(tf.transpose(mid, perm=[1, 0, 2]),
                        tf.transpose(available_centroids, perm=[0, 2, 1]))  # (D, bs, K)
        response = -norm_1 + 2 * tf.transpose(dot, perm=[1, 0, 2]) - norm_2
        response = tf.reshape(response, [-1, self.d, self.k//2])
        response = self.batch_norm2(response, training=training)
        codes_mid = tf.argmax(response, -1, output_type=tf.int32)


        available_centroids = tf.slice(self.centroids_k, [0, 0, 0], [-1, self.k//8, -1])

        tail = tf.reshape(tail, [-1, self.d, self.sub_size])
        norm_1 = tf.math.reduce_sum(tail**2, -1, keepdims=True)  # (bs, D, 1)
        norm_2 = tf.expand_dims(tf.reduce_sum(available_centroids**2, -1), 0)  # (1, D, K)
        dot = tf.matmul(tf.transpose(tail, perm=[1, 0, 2]),
                        tf.transpose(available_centroids, perm=[0, 2, 1]))  # (D, bs, K)
        response = -norm_1 + 2 * tf.transpose(dot, perm=[1, 0, 2]) - norm_2
        response = tf.reshape(response, [-1, self.d, self.k//8])
        response = self.batch_norm3(response, training=training)
        codes_tails = tf.argmax(response, -1, output_type=tf.int32)

        codes = tf.dynamic_stitch(condition_indices, [codes_head, codes_mid, codes_tails])
        # combined = tf.concat([codes_head, codes_mid, codes_tails], 0)
        # order = tf.reshape(tf.argsort(tf.concat([inputs_head_ids, inputs_mid_ids, inputs_tail_ids], 0), 0), [-1])
        # codes = tf.gather(combined, order)

        # Compute the codes based on response.

        neighbor_idxs = codes

        # Compute the outputs, which has shape (batch_size, D, d_out)
        if not self.shared_centroids:
            D_base = tf.convert_to_tensor(
                [self.k * d for d in range(self.d)], dtype=tf.int32)
            neighbor_idxs += tf.expand_dims(D_base, 0)  # (batch_size, D)
        neighbor_idxs = tf.reshape(neighbor_idxs, [-1])  # (batch_size * D)
        centroids_k = tf.reshape(self.centroids_k, [-1, self.sub_size])
        
        outputs = tf.nn.embedding_lookup(centroids_k, neighbor_idxs)
        outputs = tf.reshape(outputs, [-1, self.d, self.sub_size])
        outputs_final = tf.stop_gradient(outputs - inputs) + inputs
        
        # Add regularization for updating centroids / stabilization.
        if training:
            alpha = 1.
            beta = self.beta
            gamma = 0.0
            reg = alpha * tf.reduce_mean(
                (outputs - tf.stop_gradient(inputs))**2, name="centroids_adjust")
            reg += beta * tf.reduce_mean(
                (tf.stop_gradient(outputs) - inputs)**2, name="input_commit")
            minaxis = [0, 1] if self.shared_centroids else [0]
            reg += gamma * tf.reduce_mean(  # could sg(inputs), but still not eff.
                tf.reduce_min(-response, minaxis), name="de_isolation")
        
            self.add_loss(reg)
        

        return codes, outputs_final

if __name__ == "__main__":
    # VQ
    freq = pd.DataFrame({'id': [0, 1, 2],'freq': [1, 95, 100]})
    kdq_demo = TripleMGQEEmbedding(16, 4, 3, 4, freq, False)
    #kdq_demo = DPQEmbedding(8, 4, 2, 4, False)
    result = kdq_demo(tf.concat([tf.fill([3], 0), tf.fill([3], 1), tf.fill([3], 2)], axis=0), training=False)
    print(result)
    print("vars:")
    print(kdq_demo.trainable_variables)
