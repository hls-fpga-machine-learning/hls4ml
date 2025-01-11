import numpy as np
import tensorflow as tf

from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES


@tf.keras.utils.register_keras_serializable(name='DenseRegularizer')
class DenseRegularizer(tf.keras.regularizers.Regularizer):
    '''
    A flexible regularizer for Dense layers, simultaneously penalizing high values and variance

    Args:
        alpha (float): Sparse penalty; a higher value pushes more weights towards zero
        beta (float): Variance penalty; a higher value reduces variance between a group of weights
        norm (int): Norm type (l1 or l2)
        structure_type (string): Type of regularization - unstructured, structured, pattern, block
        block_shape (tuple): Block shape if structure_type == block
        pattern_offset (int): Length of each pattern if structure_type == pattern
        consecutive_patterns (int): How many consecutive patterns should be considered
        weights (tf.Variable): Two-dimensional layer weight tensor, dimensionality (M x N)

    Returns:
        Regularizer penalty (tf.Variable): Penalty associated with layer weights

    Examples:
        - structure_type = unstructured: unstructured weight regularization
        - structure_type = structured: neuron regularization
            (group weights by row)
        - structure_type = pattern: regularization on groups of every n-th weight
            (e.g. grouping by reuse factor in hls4ml)
        - structure_type = block: regularization on blocks within weight matrix
            (e.g. 4x4, 8x1 for certain SIMD processors)

        - consecutive_patterns is commonly encountered with optimization of BRAM utilization -
            e.g. while it is true that each DSP pattern consumes one DSP,
            They likely use less than one BRAM block (e.g. if the BRAM width is 36 bit and weight width is 16)
            In that case, we need to group several patterns together,
            So the entire block of patterns can be removed, thus saving DSP and BRAM
    '''

    def __init__(
        self,
        alpha,
        beta=0,
        norm=1,
        structure_type=SUPPORTED_STRUCTURES.UNSTRUCTURED,
        block_shape=(1, 1),
        pattern_offset=1,
        consecutive_patterns=1,
    ):
        if norm != 1 and norm != 2:
            raise Exception(f'{self.__class__.__name__} currently supports l1- and l2-based regularization')

        if isinstance(structure_type, str):
            structure_type = SUPPORTED_STRUCTURES(structure_type)

        if not isinstance(structure_type, SUPPORTED_STRUCTURES):
            raise Exception(f'{self.__class__.__name__} unknown regularization type')

        self.alpha = alpha
        self.beta = beta
        self.norm = norm
        self.structure_type = structure_type
        self.block_shape = block_shape
        self.pattern_offset = pattern_offset
        self.consecutive_patterns = consecutive_patterns

    @tf.function
    def __call__(self, weights):
        if self.structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
            sparse_penalty = self.alpha * tf.norm(weights, ord=self.norm)
            variance_penalty = self.beta * tf.math.reduce_variance(weights)
            return sparse_penalty + variance_penalty

        if self.structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
            sparse_penalty = self.alpha * tf.norm(tf.norm(weights, axis=0, ord=2), ord=self.norm)
            variance_penalty = self.beta * tf.norm(tf.math.reduce_variance(weights, axis=0), ord=self.norm)
            return sparse_penalty + variance_penalty

        if self.structure_type == SUPPORTED_STRUCTURES.PATTERN:
            # This is equivalent to penalizing all the weights processed by the same DSP block in hls4ml.
            # The matrix is transposed, according to Resource strategy and reshaped into (pattern_offset, pattern_number)
            # Pattern offset corresponds to the number of patterns is equivalent to RF
            if (np.prod(weights.shape)) % self.pattern_offset != 0:
                tf.print(np.prod(weights.shape), self.pattern_offset)
                raise Exception(f'{self.__class__.__name__}: pattern offset needs to be a factor of matrix size')

            if self.pattern_offset % self.consecutive_patterns != 0:
                raise Exception(f'{self.__class__.__name__}: consecutive patterns need to be a factor of pattern offset')

            # Reshape weight matrix into [number_of_patterns, pattern_offset]
            number_of_patterns = np.prod(weights.shape) // self.pattern_offset
            target_shape = (self.pattern_offset, number_of_patterns)
            reshaped = tf.reshape(tf.transpose(weights), target_shape)
            # Group consecutive patterns (columns) into blocks and reshape
            # Docs for the functions to extract blocks are below [block regularization]
            total_blocks = self.pattern_offset // self.consecutive_patterns
            blocks = tf.reshape(
                tf.image.extract_patches(
                    tf.expand_dims(tf.expand_dims(reshaped, 2), 0),
                    [1, self.consecutive_patterns, number_of_patterns, 1],
                    [1, self.consecutive_patterns, number_of_patterns, 1],
                    [1, 1, 1, 1],
                    'SAME',
                ),
                (total_blocks, -1),
            )

            # Calculate penalty
            sparse_penalty = self.alpha * tf.norm(tf.norm(blocks, axis=1, ord=2), ord=self.norm)
            variance_penalty = self.beta * tf.norm(tf.math.reduce_variance(blocks, axis=1), ord=self.norm)
            return sparse_penalty + variance_penalty

        if self.structure_type == SUPPORTED_STRUCTURES.BLOCK:
            if (weights.shape[0] % self.block_shape[0]) != 0 or (weights.shape[1] % self.block_shape[1] != 0):
                raise Exception(f'{self.__class__.__name__}: block sizes need to be fators of weight matrix dimensions')

            # TensorFlow has a built-in method for extracting sub-tensors of given shape and stride
            # This method is commonly used to perform im2col,
            # Docs: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
            total_blocks = (weights.shape[0] * weights.shape[1]) // (self.block_shape[0] * self.block_shape[1])
            blocks = tf.reshape(
                tf.image.extract_patches(
                    tf.expand_dims(tf.expand_dims(weights, 2), 0),
                    [1, self.block_shape[0], self.block_shape[1], 1],
                    [1, self.block_shape[0], self.block_shape[1], 1],
                    [1, 1, 1, 1],
                    'SAME',
                ),
                (total_blocks, self.block_shape[0] * self.block_shape[1]),
            )

            sparse_penalty = self.alpha * tf.norm(tf.norm(blocks, axis=1, ord=2), ord=self.norm)
            variance_penalty = self.beta * tf.norm(tf.math.reduce_variance(blocks, axis=1), ord=self.norm)
            return sparse_penalty + variance_penalty

    def get_config(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'norm': self.norm,
            'structure_type': self.structure_type,
            'block_shape': self.block_shape,
            'pattern_offset': self.pattern_offset,
            'consecutive_patterns': self.consecutive_patterns,
        }


@tf.keras.utils.register_keras_serializable(name='Conv2DRegularizer')
class Conv2DRegularizer(tf.keras.regularizers.Regularizer):
    '''
    A flexible regularizer for Conv2D layers, simultaneously performing pruning and clustering

    Args:
        alpha (float): Sparse penalty; a higher value pushes more weights towards zero
        beta (float): Variance penalty; a higher value reduces variance between a group of weights
        norm (int): Norm type (l1 or l2)
        structure_type (string): Type of regularization - unstructured, structured, pattern
        pattern_offset (int): Length of each pattern if structure_type == pattern
        weights (tf.Variable): Four-dimensional layer weight tensor, dimensionality
            (filter_width x filter_height x n_chan x n_filt)

    Returns:
        Regularizer penalty (tf.Variable): Penalty associated with layer weights

    Example use cases:
        - structure_type = unstructured: unstructured weight regularization
        - structure_type = structured: filter regularization
            (group weights of dimensionality filt_width x filt_height x n_chan)
        - structure_type = pattern: regularization on groups of every n-th weight in flattened array
            (e.g. grouping by reuse factor in hls4ml)
    '''

    def __init__(
        self,
        alpha,
        beta=0,
        norm=1,
        structure_type=SUPPORTED_STRUCTURES.UNSTRUCTURED,
        pattern_offset=1,
        consecutive_patterns=1,
    ):
        if norm != 1 and norm != 2:
            raise Exception(f'{self.__class__.__name__} currently supports l1- and l2-based regularization')

        if isinstance(structure_type, str):
            structure_type = SUPPORTED_STRUCTURES(structure_type)

        if not isinstance(structure_type, SUPPORTED_STRUCTURES):
            raise Exception(f'{self.__class__.__name__} unknown regularization type')

        # Block pruning is only supported for Dense and QDense layers
        if structure_type == SUPPORTED_STRUCTURES.BLOCK:
            raise Exception('Block pruning is supported for 2-dimensional weight matrices')

        self.alpha = alpha
        self.beta = beta
        self.norm = norm
        self.structure_type = structure_type
        self.pattern_offset = pattern_offset
        self.consecutive_patterns = consecutive_patterns

    @tf.function
    def __call__(self, weights):
        if len(weights.shape) != 4:
            raise Exception(f'{self.__class__.__name__} regularizes Conv2D layers; weight matrix is not 4-dimensional')

        if self.structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
            sparse_penalty = self.alpha * tf.norm(weights, ord=self.norm)
            variance_penalty = self.beta * tf.math.reduce_variance(weights)
            return sparse_penalty + variance_penalty

        if self.structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
            sparse_penalty = self.alpha * tf.norm(
                tf.reduce_sum(tf.norm(weights, axis=(0, 1), ord='fro'), axis=0), ord=self.norm
            )
            variance_penalty = self.beta * tf.norm(tf.math.reduce_variance(weights, axis=(0, 1, 2)), ord=self.norm)
            return sparse_penalty + variance_penalty

        if self.structure_type == SUPPORTED_STRUCTURES.PATTERN:
            if (np.prod(weights.shape)) % self.pattern_offset != 0:
                raise Exception(f'{self.__class__.__name__}: pattern offset needs to be a factor of matrix size')

            if self.pattern_offset % self.consecutive_patterns != 0:
                raise Exception(f'{self.__class__.__name__}: consecutive patterns need to be a factor of pattern offset')

            number_of_patterns = np.prod(weights.shape) // self.pattern_offset
            target_shape = (self.pattern_offset, number_of_patterns)
            reshaped = tf.reshape(tf.transpose(weights, (3, 0, 1, 2)), target_shape)

            total_blocks = self.pattern_offset // self.consecutive_patterns
            blocks = tf.reshape(
                tf.image.extract_patches(
                    tf.expand_dims(tf.expand_dims(reshaped, 2), 0),
                    [1, self.consecutive_patterns, number_of_patterns, 1],
                    [1, self.consecutive_patterns, number_of_patterns, 1],
                    [1, 1, 1, 1],
                    'SAME',
                ),
                (total_blocks, -1),
            )

            sparse_penalty = self.alpha * tf.norm(tf.norm(blocks, axis=1, ord=2), ord=self.norm)
            variance_penalty = self.beta * tf.norm(tf.math.reduce_variance(blocks, axis=1), ord=self.norm)
            return sparse_penalty + variance_penalty

    def get_config(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'norm': self.norm,
            'structure_type': self.structure_type,
            'pattern_offset': self.pattern_offset,
            'consecutive_patterns': self.consecutive_patterns,
        }
