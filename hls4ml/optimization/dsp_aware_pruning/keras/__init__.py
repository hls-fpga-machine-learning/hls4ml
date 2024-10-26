import os
import time

import numpy as np
import tensorflow as tf

import hls4ml.optimization.dsp_aware_pruning.keras.utils as utils
from hls4ml.optimization.dsp_aware_pruning.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.dsp_aware_pruning.keras.builder import build_optimizable_model, remove_custom_regularizers
from hls4ml.optimization.dsp_aware_pruning.keras.config import SUPPORTED_LAYERS, SUPPORTED_METRICS, TMP_DIRECTORY
from hls4ml.optimization.dsp_aware_pruning.keras.masking import get_model_masks
from hls4ml.optimization.dsp_aware_pruning.keras.reduction import reduce_model
from hls4ml.optimization.dsp_aware_pruning.scheduler import OptimizationScheduler

default_regularization_range = np.logspace(-6, -2, num=16).tolist()


def optimize_model(
    model,
    model_attributes,
    objective,
    scheduler,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    epochs,
    optimizer,
    loss_fn,
    validation_metric,
    increasing,
    rtol,
    callbacks=None,
    ranking_metric='l1',
    local=False,
    verbose=False,
    rewinding_epochs=1,
    cutoff_bad_trials=1,
    directory=TMP_DIRECTORY,
    tuner='Bayesian',
    knapsack_solver='CBC_MIP',
    regularization_range=default_regularization_range,
):
    '''
    Top-level function for optimizing a Keras model, given objectives

    Args:
        model (keras.Model): Model to be optimized
        model_attributes (dict): Layer-wise model attributes,
            obtained from hls4ml.optimization.get_attributes_from_keras_model(...)
        objective (hls4ml.optimization.objectives.ObjectiveEstimator):
            Parameter, hardware or user-defined objective of optimization
        scheduler (hls4ml.optimization.scheduler.OptimizationScheduler):
            Sparsity scheduler, choose between constant, polynomial and binary
        X_train (np.array): Training inputs
        y_train (np.array): Training labels
        X_val (np.array): Validation inputs
        y_val (np.array): Validation labels
        batch_size (int): Batch size during training
        epochs (int): Maximum number of epochs to fine-tune model, in one iteration of pruning
        optimizer (keras.optimizers.Optimizer or equivalent-string description): Optimizer used during training
        loss_fn (keras.losses.Loss or equivalent loss description): Loss function used during training
        validation_metric (keras.metrics.Metric or equivalent loss description): Validation metric, used as a baseline
        increasing (boolean): If the metric improves with increased values;
            e.g. accuracy -> increasing = True, MSE -> increasing = False
        rtol (float): Relative tolerance;
            pruning stops when pruned_validation_metric < (or >) rtol * baseline_validation_metric
        callbacks (list of keras.callbacks.Callback) Currently not supported, developed in future versions
        ranking_metric (string): Metric used for ranking weights and structures;
            currently supported l1, l2, saliency and Oracle
        local (boolean): Layer-wise or global pruning
        verbose (boolean): Display debug logs during model optimization
        rewinding_epochs (int): Number of epochs to retrain model without weight freezing,
            allows regrowth of previously pruned weights
        cutoff_bad_trials (int): After how many bad trials (performance below threshold),
            should model pruning / weight sharing stop
        directory (string): Directory to store temporary results
        tuner (str): Tuning algorithm, choose between Bayesian, Hyperband and None
        knapsack_solver (str): Algorithm to solve Knapsack problem when optimizing;
            default usually works well; for very large networks, greedy algorithm might be more suitable
        regularization_range (list): List of suitable hyperparameters for weight decay

    Returns:
        keras.Model: Optimized model
    '''

    if not isinstance(scheduler, OptimizationScheduler):
        raise Exception(
            'Scheduler must be an instance of from hls4ml.optimization.scheduler.OptimizationScheduler'
            'If you provided string description (e.g. \'constant\')'
            'Please use an object instance (i.e. ConstantScheduler()).'
            'For a full list of supported schedulers, refer to hls4ml.optimization.scheduler.'
        )

    if epochs <= rewinding_epochs:
        raise Exception(
            'Please increase the number of epochs. \
                       The current epoch number is too small to perform effective pruning & weight rewinding'
        )

    if ranking_metric not in SUPPORTED_METRICS:
        raise Exception('Unknown metric for ranking weights')

    # Loss function needs to be converted to a function, string description cannot be used during custom training loop
    if isinstance(loss_fn, str):
        loss_fn = tf.keras.losses.get(loss_fn)

    # Split data set into batches
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    validation_dataset = validation_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Evaluate baseline performance
    # Use built-in function, and return as list - the metric is the second element (loss if first)
    model.compile(optimizer, loss_fn, metrics=[validation_metric])
    baseline_performance = model.evaluate(validation_dataset, verbose=0, return_dict=False)[-1]
    if verbose:
        print(f'Baseline performance on validation set: {baseline_performance}')

    # Save best weights
    # Always save weights to a file, to reduce memory utilization
    if not os.path.isdir(directory):
        os.mkdir(directory)
    if not os.path.isdir(f'{directory}/optimization'):
        os.mkdir(f'{directory}/optimization')
    model.save_weights(f'{directory}/optimization/best_weights.h5')

    # Identify optimizable layers, given the current objective
    last_optimizable_layer = utils.get_last_layer_with_weights(model)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, SUPPORTED_LAYERS):
            optimizable, optimization_attributes = objective.is_layer_optimizable(model_attributes[layer.name])
            model_attributes[layer.name].optimizable = optimizable
            model_attributes[layer.name].optimization_attributes = optimization_attributes

            # In the last layer, structured pruning can't be applied, as it removes output labels
            # Weight sharing, as well as all other types of pruning (unstructured, block etc.) are applicable
            if (
                i >= last_optimizable_layer
                and optimization_attributes.structure_type == SUPPORTED_STRUCTURES.STRUCTURED
                and optimization_attributes.pruning
            ):
                model_attributes[layer.name].optimization_attributes.pruning = False
                model_attributes[layer.name].optimizable = model_attributes[
                    layer.name
                ].optimization_attributes.weight_sharing
        else:
            model_attributes[layer.name].optimizable = False
            model_attributes[layer.name].optimization_attributes = None

    # Add regularization loss to optimizable layers
    optimizable_model = build_optimizable_model(
        model,
        model_attributes,
        optimizer,
        loss_fn,
        validation_metric,
        increasing,
        train_dataset,
        validation_dataset,
        batch_size,
        epochs // 2,
        verbose=verbose,
        directory=directory,
        tuner=tuner,
        regularization_range=regularization_range,
    )

    # Create class for masked backprop (weight freezing)
    masked_backprop = MaskedBackprop(optimizable_model, loss_fn, model_attributes)

    # In certain cases, the model might underperform at the current sparsity level, but perform better at a higher sparsity
    # Therefore, monitor the models performance over several sparsity levels and
    # Only stop pruning after high loss over several trials
    bad_trials = 0
    sparsity_conditions = True
    target_sparsity = scheduler.get_sparsity()

    while sparsity_conditions:
        # TODO - This might cause OOM issues on large models / data sets, since it is not done in batches
        gradients = (
            utils.get_model_gradients(optimizable_model, loss_fn, X_train, y_train) if ranking_metric == 'gradients' else {}
        )
        hessians = (
            utils.get_model_hessians(optimizable_model, loss_fn, X_train, y_train) if ranking_metric == 'saliency' else {}
        )

        # Mask weights
        masks, offsets = get_model_masks(
            optimizable_model,
            model_attributes,
            target_sparsity,
            objective,
            metric=ranking_metric,
            local=local,
            gradients=gradients,
            hessians=hessians,
            knapsack_solver=knapsack_solver,
        )
        for layer in optimizable_model.layers:
            if isinstance(layer, SUPPORTED_LAYERS) and model_attributes[layer.name].optimizable:
                layer_weights = layer.get_weights()
                layer_weights[0] = np.multiply(layer_weights[0], masks[layer.name]) + offsets[layer.name]
                layer.set_weights(layer_weights)

        # Mask gradients
        # Before training the model at the next sparsity level, reset internal states
        # Furthermore, modern optimizers (e.g. Adam) accumulate gradients during backprop
        # Therefore, even if the gradient for a weight is zero, it might be updated, due to previous gradients
        # Avoid this by resetting the internal variables of an optimizer
        optimizable_model.reset_metrics()
        optimizable_model.reset_states()
        for x in optimizable_model.optimizer.variables():
            x.assign(tf.zeros_like(x))
        masked_backprop.update_masks(masks)

        # Train model with weight freezing [pruning]
        if verbose:
            print(f'Pruning with a target sparsity of {target_sparsity * 100.0}% [relative to objective]')
        for epoch in range(epochs - rewinding_epochs):
            start_time = time.time()
            epoch_loss_avg = tf.keras.metrics.Mean()

            # Masked backprop
            for X, y in train_dataset:
                loss_value = masked_backprop(tf.convert_to_tensor(X), tf.convert_to_tensor(y), target_sparsity)
                epoch_loss_avg.update_state(loss_value)

            # Evaluate on validation set and print epoch summary
            if verbose:
                val_res = optimizable_model.evaluate(validation_dataset, verbose=0, return_dict=False)
                t = time.time() - start_time
                avg_loss = round(epoch_loss_avg.result(), 3)
                print(f'Epoch: {epoch + 1} - Time: {t}s - Average training loss: {avg_loss}')
                print(f'Epoch: {epoch + 1} - learning_rate: {optimizable_model.optimizer.learning_rate.numpy()}')
                print(f'Epoch: {epoch + 1} - Validation loss: {val_res[0]} - Performance on validation set: {val_res[1]}')

        # Check if model works after pruning
        pruned_performance = optimizable_model.evaluate(validation_dataset, verbose=0, return_dict=False)[-1]
        if verbose:
            print(f'Optimized model performance on validation set, after fine-tuning: {pruned_performance}')

        if __compare__(pruned_performance, rtol * baseline_performance, not increasing):
            bad_trials = 0
            sparsity_conditions, target_sparsity = scheduler.update_step()
            optimizable_model.save_weights(f'{directory}/optimization/best_weights.h5')
        else:
            bad_trials += 1
            sparsity_conditions, target_sparsity = scheduler.repair_step()

        # If the model performed poorly over several sparsity levels, stop optimization [maximum sparsity reached]
        if bad_trials > cutoff_bad_trials:
            break

        # Train model without weight freezing [rewinding]
        if verbose:
            print(f'Starting weight rewinding for {rewinding_epochs} epochs')
        optimizable_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            batch_size=batch_size,
            epochs=rewinding_epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

    # Load best weights
    optimizable_model.load_weights(f'{directory}/optimization/best_weights.h5')

    # Remove regularizers and save best model
    optimizable_model = remove_custom_regularizers(optimizable_model)
    optimizable_model.compile(optimizer, loss_fn, metrics=[validation_metric])

    # In GPU FLOP Optimization, remove structures to achieve speed-up & fine-tune the smaller architecture
    # TODO - Extend for Resource strategy in hls4ml FF optimisation
    if objective.__name__ in ('GPUFLOPEstimator'):
        optimizable_model = reduce_model(optimizable_model)
        optimizable_model.compile(optimizer, loss_fn, metrics=[validation_metric])
        optimizable_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            batch_size=batch_size,
            epochs=int(1.5 * epochs),
            callbacks=callbacks,
        )

    # Evaluate final optimized model [purely for debugging / informative purposes]
    if verbose:
        pruned_performance = optimizable_model.evaluate(validation_dataset, verbose=0, return_dict=False)[-1]
        print(f'Optimized model performance on validation set: {pruned_performance}')

    return optimizable_model


class MaskedBackprop:
    '''
    A helper class to perform masked backprop (training with frozen weights)
    The important function is __call__ as it masks gradients, based on frozen weights
    While this function can exist without a class, taking masks as input would deplete memory
    Since a new graph is created for every call, causing a large run-time
    The trick is to set the masks, models etc. as class variables and then pass the sparsity
    As the sparsity changes, a new graph of the function is created
    '''

    def __init__(self, model, loss_fn, attributes):
        self.model = model
        self.loss_fn = loss_fn
        self.attributes = attributes
        self.masks = {}

    def update_masks(self, masks):
        self.masks = masks

    @tf.function
    def __call__(self, X, y, s):
        '''
        Helper function performing backprop

        Args:
            - X (tf.Tensor): Input data
            - y (tf.Tensor): Output data
            - s (float): Sparsity

        Returns:
            - loss (tf.Varilable): Model loss with input X and output y
        '''
        grads = []
        with tf.GradientTape(persistent=True) as tape:
            output = self.model(X, training=True)
            loss = self.loss_fn(y, output)
            loss += tf.add_n(self.model.losses)
            for layer in self.model.layers:
                if layer.trainable_weights:
                    grad = tape.gradient(loss, layer.trainable_weights)
                    if self.attributes[layer.name].optimizable:
                        grad[0] = tf.multiply(grad[0], self.masks[layer.name])
                    grads += grad
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss


def __compare__(x, y, leq=False):
    '''
    Helper function for comparing two values, x & y
    Sometimes, we use the >= sign - e.g. pruned_accuracy >= tolerance * baseline_accuracy [ 0 <= tolerance <= 1]
    Other times, use the <= sign - e.g. pruned_mse <= tolerance * baseline_mse [tolerance >= 1]
    '''
    if leq:
        return x <= y
    else:
        return x >= y
