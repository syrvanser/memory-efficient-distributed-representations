import tensorflow as tf
import numpy as np

class DPQEmbedding(tf.keras.layers.Layer):

    def __init__(self, k, d, vocab_size, embebdding_size, share_subspace=False, **kwargs):
        super(DPQEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embebdding_size
        self.share_subspace = share_subspace
        self.k = k
        self.d = d
        self.sub_size = embebdding_size//d

    def call(self, inputs, training=None): # was D * subs_size before TODO
        query_wemb = self.add_weight(shape=[self.vocab_size, self.embedding_size], dtype=tf.float32, initializer="random_normal", trainable=True)
        idxs = tf.reshape(inputs, [-1]) #flatten 

        input_emb = tf.nn.embedding_lookup(query_wemb, np.asarray(idxs, dtype=np.int32)) 

        kdq = KDQuantizer(self.k, self.d, self.sub_size, self.share_subspace)

        _, input_emb = kdq(tf.reshape(input_emb, [-1, self.d, self.sub_size]), training=training)
        final_size = tf.concat([tf.shape(inputs), tf.constant([self.embedding_size])], 0)
        input_emb = tf.reshape(input_emb, final_size)
        return input_emb

class KDQuantizer(tf.keras.layers.Layer):
    def __init__(self, k, d, dim_size, shared_centroids=False, beta=0., tau=1.0, **kwargs):
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
        self.dim_size = dim_size
        self.shared_centroids = shared_centroids
        self.beta = beta
        self.tau = tf.constant(tau)

        # Create centroids for keys and values.
        d_to_create = 1 if shared_centroids else d
        self.centroids_k = self.add_weight(shape=[d_to_create, k, dim_size], initializer="random_normal", trainable = True)
        
        if shared_centroids:
            self.centroids_k = tf.tile(self.centroids_k, [d, 1, 1])

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
        response = tf.keras.layers.BatchNormalization(scale=False, center=False)(response, training=training)

        # Compute the codes based on response.
        codes = tf.argmax(response, -1)  # (batch_size, D)
        neighbor_idxs = codes

        # Compute the outputs, which has shape (batch_size, D, d_out)
        if not self.shared_centroids:
            D_base = tf.convert_to_tensor(
                [self.k * d for d in range(self.d)], dtype=tf.int64)
            neighbor_idxs += tf.expand_dims(D_base, 0)  # (batch_size, D)
        neighbor_idxs = tf.reshape(neighbor_idxs, [-1])  # (batch_size * D)
        centroids_k = tf.reshape(self.centroids_k, [-1, self.dim_size])
        outputs = tf.nn.embedding_lookup(centroids_k, neighbor_idxs)
        outputs = tf.reshape(outputs, [-1, self.d, self.dim_size])
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
        
            # entropy regularization
            # reg = - beta * tf.reduce_mean(
            #    tf.reduce_sum(response_prob * safer_log(response_prob), [2]))
            self.add_loss(reg)

        return codes, outputs_final

if __name__ == "__main__":
    # VQ
    kdq_demo = DPQEmbedding(4, 8, 50, 64, True)
    result = kdq_demo(tf.zeros([1, 100]), training=True)
    print(result)
    print("vars:")
    print(kdq_demo.trainable_variables)
