import os
import math
import time
from datetime import timedelta

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from keras import backend as K

from sklearn.model_selection import KFold
from sklearn.utils import check_random_state

import cv2
from PIL import Image
from perlin_noise import PerlinNoise

from tqdm import tqdm
#======================================================================================
from unetseg import Unet, BatchModulation
from ContinuousDatasets import *
#======================================================================================
#os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
#======================================================================================
BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 5

CLASSES = ['metastasis']
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
#======================================================================================
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss# + focal_loss
    
metrics = [dice_loss, sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), sm.metrics.Precision(), sm.metrics.Recall(), tf.keras.metrics.AUC()]
#======================================================================================
def build_new_model2(styled=False):
    input_shape = (256, 256, 3)
    
    outmodels = Unet(input_shape, styled)
    if styled:
        model2 = outmodels[0]
        smodel = outmodels[1]
    else:
        model2 = outmodels
    
    optim = tf.keras.optimizers.Adam()#0.0001)#0.00001)
    
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()  #if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    ce_loss = sm.losses.BinaryCELoss()
    #ce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    total_loss = dice_loss# + focal_loss# + ce_loss
    #total_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    metrics = [dice_loss, sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), tf.keras.metrics.AUC()]
    
    model2.compile(optimizer=optim, loss=total_loss, metrics=metrics)

    if not styled:
        return model2

    optim2 = tf.keras.optimizers.Adam()
    sloss = keras.losses.SparseCategoricalCrossentropy()
    
    smodel.compile(optimizer=optim2, loss=sloss, metrics=[tf.keras.metrics.AUC()])
    
    return model2, smodel
#======================================================================================
def clone_model(model):
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    return new_model
    
def evaluate_tsk(model, task_list):
    #mean_metric = 0.0
    #mean_metric2 = 0.0
    mets = []
    for task in task_list:
        if type(task) is tuple:
            evals = model.evaluate(task[1], verbose=2)
            mets.append(evals)
            #mean_metric += evals[0]
            #mean_metric2 += evals[-1]
        else:
            evals = model.evaluate(task, verbose=2)
            mets.append(evals)
            #mean_metric += evals[0]
            #mean_metric2 += evals[-1]
    return mets
    #print('Mean 1st metric: ', mean_metric/len(task_list))
    #print('Mean last metric: ', mean_metric2/len(task_list))
#======================================================================================
all_types = ['Adrenal_gland','Bile-duct','Bladder','Breast','Cervix','Colon','Esophagus','HeadNeck','Kidney','Liver','Lung','Ovarian','Pancreatic','Prostate','Skin','Stomach','Testis','Thyroid','Uterus']
print(len(all_types))

for i in range(len(all_types)):
    type_cancer = all_types[i]
    print(i, type_cancer, len(os.listdir('Cancer_Segmentation/'+type_cancer)))
#======================================================================================
#lyn_train_ids, lyn_val_ids = split_lynsec(split=0.5)

#lyn_train = DataloderLynSec(ids=lyn_train_ids, batch_size=BATCH_SIZE, shuffle=True)
#lyn_val = DataloderLynSec(ids=lyn_val_ids, batch_size=BATCH_SIZE, shuffle=True)
#print('LYN', len(lyn_train), len(lyn_val))
#======================================================================================
def compute_MAS_matrix(model, loss, ds_train):
    avg_delta = None

    #mseloss = tf.keras.losses.MSE()#(y_true, y_pred)
    
    for x, y in tqdm(ds_train):
        y = tf.Variable(y, dtype='float32')
        with tf.GradientTape(persistent=True) as tape:
            logits = model(x, training=True)
            # tf.math.reduce_max(logits, axis=1)
            #loss_value = tf.keras.losses.MSE(tf.zeros(logits.shape), logits)
            
            #loss_value = tf.math.reduce_mean(tf.math.pow(tf.math.l2_normalize(logits), tf.constant(2.0)))
            #loss_value = tf.math.reduce_mean(tf.math.pow(logits, tf.constant(2.0)))
            loss_value = tf.math.reduce_mean(tf.norm(tf.concat(logits,1), 2, 1))

        grads = tape.gradient(loss_value, model.trainable_weights)
        del tape

        for g in range(len(grads)):
            if grads[g] is None:
                grads[g] = np.zeros(model.trainable_weights[g].shape)
        
        if avg_delta is None:
            avg_delta = []
            for i in range(len(grads)):
                avg_delta.append(np.absolute(grads[i]))
        else:
            for i in range(len(grads)):
                avg_delta[i] = avg_delta[i] + np.absolute(grads[i])
    
    for i in range(len(avg_delta)):
        avg_delta[i] = avg_delta[i] / len(ds_train)
    
    return avg_delta

def MAS_loss(model, loss, fim, prev_params, lambd): 
    
    def custom_loss(y_true, y_pred):
        normal_loss=loss(y_true,y_pred)

        regularization_value=tf.constant([0.])
        for layer in range(len(fim)):
            regularization_value += tf.reduce_sum(fim[layer] * tf.math.pow(tf.math.abs(prev_params[layer]-model.trainable_variables[layer]), tf.constant(2.0)))

        return normal_loss+regularization_value * lambd
    
    return custom_loss
#======================================================================================
def compute_fisher_matrix(model, task_set, batch_size):
  # Build fisher matrixes dictionary: at each key it will store the Fisher matrix for a particular layer
  fisher_matrixes = {n: tf.zeros_like(p.value()) for n, p in enumerate(model.trainable_variables)}
    
  #for i, (imgs, labels) in enumerate(task_set.take(batch_size)):
  for imgs, labels in task_set:#tqdm(task_set):
    # Initialize gradients storage
    with tf.GradientTape() as tape:
      # Compute the predictions (recall: we will just take the prediction from the head related to the actual task)
      #preds = model(imgs)[task_id]
      preds = model(imgs)

      # Compute the logarithm of the predictions
      ll= tf.math.log(preds)

    # Attach gradients over the log_likelihood to log_likelihood_grads
    ll_grads  = tape.gradient(ll, model.trainable_variables)
      
    # Compute Fisher matrix at each layer (if existing)
    for i, gradients in enumerate(ll_grads):
        if gradients != None:
            fisher_matrixes[i] += tf.math.reduce_mean(gradients ** 2, axis=0) / batch_size
  return fisher_matrixes

def EWC_loss(model, loss, fim, prev_params, lambd):

    def custom_loss(y_true, y_pred):
        normal_loss=loss(y_true, y_pred)

        regularization = tf.constant([0.])
        for layer in range(len(fim)):
            regularization += tf.reduce_sum(fim[layer]*(prev_params[layer]-model.trainable_variables[layer])**2)

        # Return the standard cross entropy loss + EWC regularization
        return normal_loss + regularization * lambd
    
    return custom_loss
#======================================================================================
def fit_multi_segmentation(train, val, batch_size, model, loss, optimizer, metrics, epochs, fim, prev_params, lambd=100, ewc=1):
    loss_fn = loss
    if fim is not None:
        if ewc == 0:
            loss_fn = MAS_loss(model, loss, fim, prev_params, lambd)
        elif ewc == 1:
            loss_fn = EWC_loss(model, loss, fim, prev_params, lambd)
        elif ewc == 2:
            loss_fn = EWC2_loss(model, loss, fim, prev_params)
    
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    #model.fit(train, validation_data=val, epochs=epochs)
    model.fit(
        train,
        steps_per_epoch=len(train), 
        epochs=epochs, 
        validation_data=val, 
        validation_steps=len(val),
        verbose=2
    )

    prev_params = [tf.identity(model.trainable_variables[layer]) for layer in range(len(model.trainable_variables))]

    if ewc == 0:
        new_fim = compute_MAS_matrix(model, loss, val)
    elif ewc == 1:
        new_fim = compute_fisher_matrix(model, val, batch_size=batch_size)
    elif ewc == 2:
        new_fim = compute_EWC2_matrix(model, loss, val)
    
    if fim is None:
        fim = new_fim.copy()
    else:
        if ewc == 0:
            for layer in range(len(fim)):
                #fim[layer] = (0.5 * fim[layer]) + (0.5 * new_fim[layer])
                fim[layer] = fim[layer] + new_fim[layer]
        else:
            for layer in range(len(fim)):
                fim[layer] = fim[layer] + new_fim[layer]

    return fim, prev_params
#======================================================================================
def regularization_loss(model, loss, importance, old_vars, lambd=1):
    def custom_loss(y_true, y_pred):
        loss_value = loss(y_true, y_pred)
        reg_loss = tf.constant([0.])
        for l in range(len(importance)):
            reg_loss += tf.reduce_sum(
                importance[l] * tf.math.pow(tf.math.abs(old_vars[l]-model.trainable_weights[l]), tf.constant([2.0]))#importance[l] * 
                #importance[l] * tf.math.pow(tf.math.abs(old_vars[l]-model.weights[l]), tf.constant([2.0]))
                #importance[l] * tf.math.abs(old_vars[l]-model.weights[l])
            )
        #print(reg_loss)
        return loss_value+(reg_loss*lambd)
    return custom_loss
#======================================================================================
def importance_matrix(model, loss, task_data):
    avg_delta = None
    if task_data is None:
        avg_delta = []
        for w in range(len(model.trainable_weights)):
            avg_delta.append(np.zeros(model.trainable_weights[w].shape).astype(float))
        return avg_delta

    for x, y in task_data:
        y = tf.Variable(y, dtype='float32')
        with tf.GradientTape(persistent=True) as tape:
            preds = model(x, training=True)
            #loss_value = tf.math.pow(tf.math.log(loss(y, preds)), tf.constant(2.0))
            #loss_value = loss(y, preds)
            
            #loss_value = tf.math.log(loss(y, preds))
            loss_value = tf.math.reduce_mean(tf.norm(tf.concat(preds,1), 2, 1))
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        del tape
        
        if avg_delta is None:
            avg_delta = grads.copy()
            for g in range(len(grads)):
                if 'batch_modulation' in model.trainable_weights[g].name:
                    avg_delta[g] = np.zeros(model.trainable_weights[g].shape).astype(float)
                    continue
                if grads[g] is not None:
                    #avg_delta[g] = np.abs(grads[g]) * 1000000.0
                    avg_delta[g] = np.abs(grads[g])
        else:
            for g in range(len(grads)):
                if 'batch_modulation' in model.trainable_weights[g].name:
                    continue
                if grads[g] is not None:
                    #avg_delta[g] = avg_delta[g] + (np.abs(grads[g]) * 1000000.0)
                    avg_delta[g] = avg_delta[g] + np.abs(grads[g])

    importance = avg_delta.copy()
    for i in range(len(avg_delta)):
        importance[i] = avg_delta[i] / len(task_data)
        if tf.is_tensor(importance[i]):
            importance[i] = importance[i].numpy()

    #return importance

    full_importance = []
    index_imp = 0
    n_jumps = 0
    for w in range(len(model.weights)):
        if n_jumps > 0:
            n_jumps -= 1
            #print(w, model.weights[w].name, 'jump')
            continue
        #print(w, model.weights[w].name)
            
        if model.weights[w].trainable:
            if ('Conv' in model.weights[w].name) and True:
                #print(index_imp, importance[index_imp].shape, w, model.weights[w].name)
                max_per_neuron = np.max([
                    np.max(importance[index_imp], axis=(0,1,2)),
                    importance[index_imp+1]
                ], axis=0)
                
                #ws = [np.ones(importance[index_imp].shape)[:,:,:,j] * max_per_neuron[j] for j in range(importance[index_imp].shape[-1])]
                ws = np.ones(importance[index_imp].shape) * max_per_neuron
                bs = max_per_neuron
                
                full_importance.append(ws)
                full_importance.append(bs)
                index_imp = index_imp + 2
                n_jumps = 1
            elif (('BN' in model.weights[w].name)):# and ('gamma' in model.weights[w].name)) and True:
                #print(importance[index_imp].shape, importance[index_imp+1].shape, importance[index_imp+2].shape, importance[index_imp+3].shape)
                max_per_neuron = np.max([importance[index_imp], importance[index_imp+1]], axis=0)#, importance[index_imp+2], importance[index_imp+3]], axis=0)
                #sum_per_neuron = importance[index_imp] + importance[index_imp-1]
                
                full_importance.append(max_per_neuron)
                full_importance.append(max_per_neuron)
                #full_importance.append(max_per_neuron)
                #full_importance.append(max_per_neuron)
                
                index_imp = index_imp + 2
                #index_imp = index_imp + 1
                n_jumps = 1
            else:
                full_importance.append(importance[index_imp])
                index_imp = index_imp + 1
        else:
            continue
            #full_importance.append(np.zeros(model.weights[w].shape).astype(float))
            
    return full_importance#importance
#======================================================================================
def merge_model_weights(models, importances):
    teoric_weights = []
    for i in range(len(models[0])):
        weights = []
        for j in range(len(models)):
            weights.append(models[j][i])
        #imps = []
        #for j in range(len(importances)):
        #    imps.append(importances[j][i])
        teoric_weights.append(weighted_mean(weights, importances))
        #teoric_weights.append(weighted_mean(weights, imps))
        #teoric_weights.append(weighted_median(weights, np.array(importances)))
    return teoric_weights
def weighted_mean(values, importances):
    sums = None
    weights = None
    for i in range(len(values)):
        #weights += importances[i]
        if weights is None:
            weights = importances[i]
        else:
            weights += importances[i]
        if sums is None:
            sums = values[i]*importances[i]
        else:
            sums += values[i]*importances[i]
    return sums / weights

def weighted_median(values, weights):
    data = list(zip(values, weights))
    #data.sort()
    sorted(data, key=lambda x: x[1]) #sort by weight

    midpoint = weights.sum() / 2.0

    cumulative_weight = 0
    for value, weight in data:
        cumulative_weight += weight
        if cumulative_weight >= midpoint:
            return value

    raise 'No_Median'
#======================================================================================
class MergeCB(keras.callbacks.Callback):
    def __init__(self, weights, importance, w_imp, n_loss_fn, n_ds):
        super().__init__()
        self.old_weights = weights
        self.old_imp_matrix = w_imp
        self.new_imp_loss_fn = n_loss_fn
        self.new_imp_dataset = n_ds
        self.old_importance = importance
        self.best_w = None
        self.best_i = None
        
    # on_epoch_end
    def on_epoch_begin(self, epoch, logs=None):
        if self.old_weights is None:
            return
        self.new_imp_matrix = importance_matrix(self.model, self.new_imp_loss_fn, self.new_imp_dataset)
    #    if self.best_w is None:
    #        self.best_w = self.model.get_weights()
    #        self.best_i = logs.get("val_loss")
    #    elif logs.get("val_loss") < self.best_i:
    #        self.best_w = self.model.get_weights()
    #        self.best_i = logs.get("val_loss")

    #def on_epoch_end(self, epoch, logs=None):
    def on_train_batch_end(self, batch, logs=None):
        if self.old_weights is None:
            return
        merge_w = [self.old_weights, self.model.get_weights()]
        #old_i = [mat+self.old_importance for mat in self.old_imp_matrix]
        #new_i = [mat+logs.get("loss") for mat in self.new_imp_matrix]
        #merge_i = [old_i, new_i]
        merge_i = [self.old_importance, logs.get("loss")]
        teoric_weights = merge_model_weights(merge_w, merge_i)
        self.model.set_weights(teoric_weights)

class TestCB(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.means = []
        self.traks = []
        self.total = []
        self.wigths = []
        #self.vars = []
        
    def on_epoch_end(self, epoch, logs=None):
        imod = 4
        self.wigths.append(self.model.layers[2].get_weights()[1].copy())
        self.means.append(self.model.layers[imod].moving_mean.numpy())
        #self.traks.append(self.model.layers[imod].track_mean.numpy())
        #self.total.append(self.model.layers[imod].total_mean.numpy())
        #if self.model.layers[imod].drifting or self.model.layers[imod].smithing:
        #    self.total.append(self.model.layers[imod].track_mean.numpy())
        #    self.vars.append(self.model.layers[3].track_variance.numpy())
        #else:
        #    self.total.append(self.model.layers[imod].moving_mean.numpy())
        #    self.vars.append(self.model.layers[3].moving_variance.numpy())
#======================================================================================
#======================================================================================
#======================================================================================
#task_list = [(t1_canseg_train, t1_canseg_val),(t3_canseg_train, t3_canseg_val),(lyn_train, lyn_val),(t6_canseg_train, t6_canseg_val)]
task_list = []

for i in range(len(all_types)):
    #if i > 2:
    #if i not in [0, 3]:
        #if i >= 8: # only the first 8 are good samples
    #    continue
    train_ids, val_ids = split_canser_seg(all_types[i:i+1], split=0.7)
    print(all_types[i:i+1])
    canseg_train = DataloderCancerSeg(ids=train_ids, batch_size=BATCH_SIZE, shuffle=True)
    canseg_val = DataloderCancerSeg(ids=val_ids, batch_size=BATCH_SIZE, shuffle=True)

    task_list.append((canseg_train, canseg_val))
    
#task_list.append((lyn_train, lyn_val))

#task_list.reverse()
#======================================================================================
#======================================================================================
#======================================================================================
run_complete = 0
run_ours = 0
run_ewc = 10
run_mas = 10
print("======================================================================================")
print("run_complete", run_complete)
print("run_ours", run_ours)
print("run_ewc", run_ewc)
print("run_mas", run_mas)
print("======================================================================================")
#======================================================================================
#======================================================================================
#======================================================================================
# Complete
if run_complete > 0:
    aux_train = []
    aux_val = []
    for train, val in task_list:
        aux_train.append(train)
        aux_val.append(val)
        
    fullds_train = JoinDataSets(datasets=aux_train, fulljoin=True, batchsize=BATCH_SIZE)
    fullds_val = JoinDataSets(datasets=aux_val, fulljoin=True, batchsize=BATCH_SIZE)
    del aux_train
    del aux_val
    
    start_total = time.time()
    # ###############################################
    total_performance = []
    for i in range(len(task_list)):
        total_performance.append([])
    
    for run_iter in range(run_complete):
        print('run_complete iteration', run_iter)
        start_iter = time.time()

        quicktest = build_new_model2(styled=False)
        quickopt = keras.optimizers.Adam(LR)
        quicktest.compile(loss=total_loss, optimizer=quickopt, metrics=metrics)#, run_eagerly=True)

        history = quicktest.fit(fullds_train, steps_per_epoch=len(fullds_train), epochs=100, validation_data=fullds_val, validation_steps=len(fullds_val), verbose=2)
        mets = evaluate_tsk(quicktest, task_list)

        #total_performance.append(mets)
        for i in range(len(total_performance)):
            total_performance[i].append(mets[i])

        elapsed_iter = (time.time() - start_iter)
        print('COMPLETE iter time', run_iter, str(timedelta(seconds=elapsed_iter)))

    elapsed = (time.time() - start_total)
    print('COMPLETE total time', str(timedelta(seconds=elapsed)))

    #print('COMPLETE task array', total_performance)
    for i in range(len(total_performance)):
        print('COMPLETE task array', i, total_performance[i])

#======================================================================================
# OURS
def diff_models(model1, model2):
    means = []
    for i in range(len(model1)):
        means.append(np.mean((model2[i] - model1[i])**2))
    return np.mean(means)

def recall_loss(y_true, y_pred):
    #print(y_true - y_pred)
    #print(y_true)
    tp = y_true * y_pred
    fn = y_true - tp
    mask = ((tp) + 0.000001) / (tp + fn + 0.000001)
    #print(mask)
    return 1 - tf.math.reduce_mean(tf.cast(mask, tf.float32))#/tf.cast(tf.size(mask), tf.float32)


if run_ours > 0:
    start_total = time.time()

    total_performance = []
    for i in range(len(task_list)):
        total_performance.append([])
    
    for run_iter in range(run_ours):
        print('run_ours iteration', run_iter)
        start_iter = time.time()

        quicktest = build_new_model2(styled=False)
        quickopt = keras.optimizers.Adam(LR)
        #testcb = TestCB()
        #testcb2 = TestCB()
        quicktest.compile(loss=total_loss, optimizer=quickopt, metrics=metrics)#, run_eagerly=True)

        join_modelo = None
        prev_w = None
        prev_i = None

        wimportance = []

        curr_task = 0

        fim = None
        prev_params = None

        histories = []
        base_hits = 0
        
        ref_recall = None
        for ds_train, ds_val in task_list:
            start_task = time.time()

            #if curr_task > 0:
                #evals = quicktest.evaluate(ds_val)
                #loss_diff = evals[0] - moving_loss
                #loss_diff = moving_loss - evals[0]
                #if loss_diff > 0:
                #    continue
                #lcalc = 10**(2+loss_diff)
                #print('new task loss', evals[0], 'moving_loss', moving_loss, 'lambda', lcalc)
                # impedir se loss for menor
                
                #prev_vars = [tf.identity(quicktest.weights[l]) for l in range(len(quicktest.weights))]
            #    prev_vars = [tf.identity(quicktest.trainable_weights[l]) for l in range(len(quicktest.trainable_weights))]
            #    loss_fn = regularization_loss(quicktest, total_loss, wimportance, prev_vars, lambd=100) # 100
            #    quicktest.compile(loss=loss_fn, optimizer=quickopt, metrics=metrics)#, run_eagerly=True)

            #new_importance = importance_matrix(quicktest, total_loss, ds_val)
            #mergcb = MergeCB(prev_w, prev_i, wimportance, total_loss, ds_val)
            if curr_task == 0:
                history = quicktest.fit(ds_train, steps_per_epoch=len(ds_train), epochs=10, validation_data=ds_val, validation_steps=len(ds_val), verbose=2)#, callbacks=[mergcb])
            else:

                #lrrecall = SetpValue(start_value=0.5, step_size=0.1)
                #ref_recall = None

                #quicktest.fit(ds_train, steps_per_epoch=len(ds_train), epochs=10, validation_data=ds_val, validation_steps=len(ds_val), verbose=2)
                #spec_w = quicktest.get_weights()
                #evaluate_tsk(quicktest, task_list)
                
                #quicktest.set_weights(prev_w)
                #quicktest.compile(loss=total_loss, optimizer=quickopt, metrics=metrics)
        
                for e in range(10):
                    #total_recall = 0
                    #tot_loss = 0
                    for x, y in ds_train:
                        y = tf.Variable(y, dtype='float32')

                        curr_w = quicktest.get_weights()
                        quicktest.set_weights(prev_w)
                        quicktest.compile(loss=total_loss, optimizer=quickopt, metrics=metrics)
                        old_preds = quicktest(x, training=False)

                        #quicktest.set_weights(spec_w)
                        #quicktest.compile(loss=total_loss, optimizer=quickopt, metrics=metrics)
                        #new_preds = quicktest(x, training=False)
                
                        quicktest.set_weights(curr_w)
                        quicktest.compile(loss=total_loss, optimizer=quickopt, metrics=metrics)
                        #del curr_w
                        #old_preds = aux_model(x, training=False)
                        
                        with tf.GradientTape(persistent=True) as tape:
                            #old_preds = aux_model(x, training=False)                
                            preds = quicktest(x, training=True)
        
                            #recal_i = lrrecall.get_val()
                            #total_recall += recall_loss(old_preds, preds)
                            #rec_val = recall_loss(old_preds, preds)
                            #neg_val = neg_loss(old_preds, preds)
                            #if ref_recall is None:
                            #    ref_recall = rec_val
                            #if ref_negative is None:
                            #    ref_negative = neg_val
                            #  aux_model.get_weights()
                            #rec_lls = diff_models(quicktest.get_weights(), prev_w) + ((1 - rec_val) - (1 - ref_recall))# + ((1 - neg_val) - (1 - ref_negative))
                            #rec_lls = 0.0 if rec_lls < 0.0 else rec_lls
                            rec_lls = total_loss(old_preds, preds)+total_loss(y*old_preds, y*old_preds*preds)
                            
                            #gt_mks = tf.cast(~(y > 0), dtype='float32')
                            #lls = total_loss(y+(gt_mks * old_preds), preds)
                            #lls = total_loss(y, preds)
                            lls = total_loss(y, preds)
        
                            #ll2 = total_loss(y, preds)
        
                            loss_value = lls+rec_lls#+ll2
                    
                            #loss_value = total_loss(y, preds*y)
                            #tot_loss += loss_value
                            #print(recal_i, loss_value)
                        grads = tape.gradient(loss_value, quicktest.trainable_weights)
                        del tape
                        quickopt.apply_gradients(zip(grads, quicktest.trainable_variables))
        
                        #print('total_loss', loss_value, 'lls', lls, 'rec_lls', rec_lls)#, 'rec_val', rec_val)
                    del preds
                    del old_preds
                    del grads
                    #print(total_loss(y, preds), recal_i, recall_loss(old_preds, preds))
                    #quicktest.evaluate(ds_val)
                    #total_recall = total_recall/len(ds_train)
                    #tot_loss = tot_loss/len(ds_train)
                    #print('tot_loss', tot_loss, 'total_recall', total_recall)
                    
                    #evaluate_tsk(quicktest, task_list)
        
                    #quicktest.set_weights(merge_model_weights(
                    #    [aux_model.get_weights(), quicktest.get_weights()], 
                    #    [prev_i, mets[curr_task][0]],
                    #    imp_matrix=False
                    #))
        
                    #if mets[curr_task][2] > 0.35:
                    #    print(' ================================ REACHED GOOD METRIC ================================')
                    #    break
        
                #if total_recall < 0.71:
                #    print(' ================================ Undoing Training ', curr_task, ' ================================')
                #    quicktest.set_weights(aux_model.get_weights())
        
                #print('rec_val < ref_recall', rec_val, '<', ref_recall, rec_val < ref_recall)
                #if rec_val < ref_recall:
                #    ref_recall = rec_val
        
                del curr_w
                #del aux_model
                print('')
                print(' ================================ Done training ', curr_task, ' ================================')
                print('')
                
            #if curr_task > 0:
            #    merge_w = [prev_w, mergcb.best_w]
            #    merge_i = [prev_i, mergcb.best_i]
            #    teoric_weights = merge_model_weights(merge_w, merge_i)
            #    quicktest.set_weights(teoric_weights)
        
            #print(quicktest.layers[33].get_weights()[1])
        
            #changeBMmodel(quicktest, 'new_style')
            
            #histories.append(history.history.copy())
            #evals = quicktest.evaluate(ds_val, verbose=2)
        
            mets = evaluate_tsk(quicktest, task_list)
            
            for i in range(len(total_performance)):
                total_performance[i].append(mets[i])
            #break
        
            #if curr_task >= 3:
            #    break
        
            #next_importance = importance_matrix(quicktest, total_loss, ds_val)
            #if curr_task == 0:
            #    wimportance = next_importance.copy()
                #moving_loss = evals[0]
            #else:
                #if len(wtrust) == 0:
                #    wtrust = [n-o for o, n in zip(wimportance, next_importance)]
                #else:
                #    wtrust = [(t*0.5)+((n-o)*0.5) for o, n, t in zip(wimportance, next_importance, wtrust)]
                    
            #    wimportance = [o + n for o, n in zip(wimportance, next_importance)]
                #wimportance = [(o + n) + (n-o) for o, n in zip(wimportance, next_importance)]
                #wimportance = [(0.5 * o) + (0.5 * n) for o, n in zip(wimportance, next_importance)]
                #wimportance = [weighted_mean(np.array([o, n]), np.array([prev_i, evals[0]])) for o, n in zip(wimportance, next_importance)]
                #moving_loss = (moving_loss * 0.5) + (evals[0] * 0.5)
        
            #if curr_task > 0:
            #    merge_w = [prev_w, quicktest.get_weights()]
            #    merge_i = [prev_i, evals[0]]
            #    teoric_weights = merge_model_weights(merge_w, merge_i)
            #    evaluate_tsk(join_modelo, task_list)
        
            prev_w = quicktest.get_weights()
            #prev_i = evals[0]
            #if prev_i is not None:
            #    prev_i = (prev_i * 0.9) + (evals[0] * 0.1)
            #    #prev_i = prev_i + (evals[0] * 0.2)
            #else:
            #    prev_i = evals[0]
        
            #if curr_task > 0:
            #    quicktest.set_weights(teoric_weights)

            elapsed_task = (time.time() - start_task)
            print('OUR task time', curr_task, str(timedelta(seconds=elapsed_task)))
            curr_task += 1

        elapsed_iter = (time.time() - start_iter)
        print('OUR iter time', run_iter, str(timedelta(seconds=elapsed_iter)))
            
    elapsed = (time.time() - start_total)
    print('OUR total time', str(timedelta(seconds=elapsed)))

    for i in range(len(total_performance)):
        print('OUR task array', i, total_performance[i])
#======================================================================================
# EWC 1
if run_ewc > 0:
    start_total = time.time()

    total_performance = []
    for i in range(len(task_list)):
        total_performance.append([])

    for run_iter in range(run_ewc):
        print('run_ewc iteration', run_iter)
        start_iter = time.time()
        
        quicktest = build_new_model2(styled=False)
        optim = keras.optimizers.Adam(LR)
        fim = None
        prev_params = None

        curr_task = 0
        for task in task_list:
            start_task = time.time()
            
            train_data, val_data = task
            fim, prev_params = fit_multi_segmentation(train_data, val_data, BATCH_SIZE, quicktest, total_loss, optim, metrics, 10, fim, prev_params, lambd=1000, ewc=1)
            mets = evaluate_tsk(quicktest, task_list)
            
            for i in range(len(total_performance)):
                total_performance[i].append(mets[i])

            elapsed_task = (time.time() - start_task)
            print('EWC task time', curr_task, str(timedelta(seconds=elapsed_task)))
            curr_task += 1

        elapsed_iter = (time.time() - start_iter)
        print('EWC iter time', run_iter, str(timedelta(seconds=elapsed_iter)))

    elapsed = (time.time() - start_total)
    print('EWC total time', str(timedelta(seconds=elapsed)))

    for i in range(len(total_performance)):
        print('EWC task array', i, total_performance[i])
#======================================================================================
# MAS
if run_mas > 0:
    start_total = time.time()

    total_performance = []
    for i in range(len(task_list)):
        total_performance.append([])
    
    for run_iter in range(run_mas):
        print('run_mas iteration', run_iter)
        start_iter = time.time()
        
        quicktest = build_new_model2(styled=False)
        optim = keras.optimizers.Adam(LR)
        fim = None
        prev_params = None

        curr_task = 0
        for task in task_list:
            start_task = time.time()
            
            train_data, val_data = task
            fim, prev_params = fit_multi_segmentation(train_data, val_data, BATCH_SIZE, quicktest, total_loss, optim, metrics, 10, fim, prev_params, lambd=1000, ewc=0)
            mets = evaluate_tsk(quicktest, task_list)

            for i in range(len(total_performance)):
                total_performance[i].append(mets[i])

            elapsed_task = (time.time() - start_task)
            print('MAS task time', curr_task, str(timedelta(seconds=elapsed_task)))
            curr_task += 1

        elapsed_iter = (time.time() - start_iter)
        print('MAS iter time', run_iter, str(timedelta(seconds=elapsed_iter)))
            
    elapsed = (time.time() - start_total)
    print('MAS total time', str(timedelta(seconds=elapsed)))

    for i in range(len(total_performance)):
        print('MAS task array', i, total_performance[i])
#======================================================================================
# conda activate segModels
# nohup python experiment.py > ExperimentResults/file.out
print("DONE")