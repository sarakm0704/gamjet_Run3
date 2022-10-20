'''
DESCRIPTION:

LAST USED:
python ANN/disCo.py --dataDir /home/AapoKossi/NNinputsFinal/ --saveDir /home/AapoKossi/NN_results/regress_lam_d5_lam_r1000_epochs_5_DLogLoss0707 --k 2 --epochs 5 -v --standardize --learning_rate 3e-4 --activation "swish,swish,swish,swish,softplus" --reduce_on_plateau --regress_mass --lam_d 5 lam_r 1000 --use_cache

python ANN/disCo.py --dataDir ~/NNinputsFull/ --saveDir ~/NN_results/ --epochs 1000 --reduce_on_plateau --standardize --activation "swish,swish,swish,swish,swish,swish,softplus" --neurons 2048,1536,1024,512,256,128,2 --regress_mass --use_cache --lam_r 1000 --lam_d 5 --optimizer adamw --batchnorm False --batchSize 2048 --learning_rate 1e-3 --inputVars "tau_pt,pt_miss,R_tau,btag_all,mt,dPhiTauNu,dPhiTauJets,dPhiNuJets,dPhiAllJets,Jets_pt,mtrue"
'''
# Base script from gitlab.cern.ch/Hplus/HiggsAnalysis keras neural networks disCo.py
#================================================================================================ 
# Imports
#================================================================================================
import ROOT
# print("=== Importing KERAS")
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.platform import tf_logging as logging
from keras.utils import io_utils
from tensorflow.keras import backend
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
import pandas
import array
import os
import shutil
#import psutil
import math
import json
import copy
import random as rn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Concatenate, Dropout#, Flatten, Input
from tensorflow.keras.losses import binary_crossentropy
#from sklearn.externals import joblib
import joblib
import subprocess
import plot
import func
import tdrstyle
from jsonWriter import JsonWriter 
import sys
import time
from datetime import datetime 
from optparse import OptionParser
import getpass
import socket
from Disco_tf import distance_corr
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.losses import Loss
import datasetUtils
#================================================================================================
# Variable definition
#================================================================================================
ss = "\033[92m"
ns = "\033[0;0m"
ts = "\033[1;34m"
hs = "\033[0;35m"   
ls = "\033[0;33m"
es = "\033[1;31m"
cs = "\033[0;44m\033[1;37m"

#================================================================================================ 
# Class Definition
#================================================================================================ 

# custom callback that reduces weight decay at the same ratio with learning rate.
class myReduceLRWDOnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = backend.get_value(self.model.optimizer.lr)
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        true_factor = new_lr / old_lr
                        new_wd = backend.get_value(self.model.optimizer.weight_decay) * true_factor
                        backend.set_value(self.model.optimizer.lr, new_lr)
                        backend.set_value(self.model.optimizer.weight_decay, new_wd)
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f'\nEpoch {epoch +1}: '
                                f'ReduceLROnPlateau reducing learning rate to {new_lr}.')
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

# custom keras layer to drop input variables according to user defined model configuration
class InputSelector(tf.keras.layers.Layer):
    def __init__(self, include, name="InputSelector", **kwargs):
        super().__init__(name=name, **kwargs)
        self.include = tf.constant(include)
        outs = tf.reduce_sum(tf.cast(self.include, tf.int64))
        self.outshape = (None, outs)

    def get_config(self):
        config = super().get_config()
        config.update(include = self.include.numpy().tolist())
        return config

    def call(self, x):
        y = tf.boolean_mask(x, self.include, axis=1)
        return tf.ensure_shape(y, self.outshape)

# custom keras layer to scale inputs to the range [-1,1]
class MinMaxScaler(tf.keras.layers.Layer):
    def __init__(self, name="MinMaxScaler", maxes = None, mins = None, **kwargs):
        super().__init__(name=name, **kwargs)
        if (maxes is None and mins is not None) or (maxes is not None and mins is None):
            msg = "Must specify either both maxes and mins or leave both to be adapted"
            raise Exception(es + msg + ns)
        self.input_maxes = None
        if maxes is not None and mins is not None:
            self.input_maxes = tf.constant(maxes)
            self.input_mins = tf.constant(mins)

    def build(self, shape):
        super().build(shape)
        if self.input_maxes is not None:
            self.maxes = self.input_maxes * tf.ones(shape[1:], dtype=self.input_maxes.dtype)
            self.mins = self.input_mins * tf.ones(shape[1:], dtype=self.input_mins.dtype)            
        
    def adapt(self, data): # compute minimum and maximum of features and set as weights
        spec = data.element_spec
        dim = len(spec.shape)
        to_reduce = list(range(dim - 1))
        first = True
        for elem in data:
            maxes = tf.math.reduce_max(elem, axis=to_reduce)
            mins = tf.math.reduce_min(elem, axis=to_reduce)
            if not first:
                maxes = tf.where(maxes > prev_maxes, maxes, prev_maxes)
                mins = tf.where(mins < prev_mins, mins, prev_mins)
            else: first = False
            prev_maxes = maxes
            prev_mins = mins
        
        # add length 1 dimensions to min and max until they are broadcastable to the input shape
        while len(maxes.shape) != dim: 
            maxes = tf.expand_dims(maxes,0)
            mins = tf.expand_dims(mins, 0)
        
        self.maxes = maxes
        self.mins = mins

    def get_config(self):
        config = super().get_config()
        try: # if the layer has been adapted, return the final state
            config.update({
                "maxes": self.maxes.numpy().tolist(),
                "mins": self.mins.numpy().tolist()
            })
        except: # otherwise just return the values used to initialize the layer
            config.update({
                "maxes": self.input_maxes,
                "mins": self.input_mins
            })
        return config
        
    @tf.function
    def call(self, x):
        return 2 * (x - self.mins) / (self.maxes - self.mins) - 1

# custom keras layer to handle variable specific preprocessing transformations
class Sanitizer(tf.keras.layers.Layer):
    input_handling = {
        "tau_pt": lambda x: tf.math.log(tf.math.log(x + 1) + 1),
        "pt_miss": lambda x: tf.math.log(tf.math.log(x + 1) + 1),
        "R_tau": lambda x: tf.math.log(x + 1),
        "bjet_pt": lambda x: tf.math.log(tf.math.log(x + 1) + 1),
        "tau_mt": lambda x: tf.math.log(tf.math.log(x + 1) + 1),
        "mt": lambda x: tf.math.log(tf.math.log(x + 1) + 1),
        "dPhiTauNu": tf.math.abs,
        "dPhitaub": tf.math.abs,
        "dPhibNu": tf.math.abs,
        "mtrue" : lambda x: tf.math.log(tf.math.log(x + 1) + 1),
        "btag": lambda x: x,
        "dPhi": tf.math.abs,
        "dPhiTauJet": tf.math.abs,
        "dPhiNuJet": tf.math.abs,
        "dPhiJets": tf.math.abs,
        "jet_pt": lambda x: tf.math.log(tf.math.log(x + 1) + 1),
    }

    def __init__(self, in_var_names, name = "input_sanitizer", trainable=False, **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.inputvars = in_var_names
        self.funcs = [self.input_handling[name] for name in self.inputvars]

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_var_names": self.inputvars
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def scale(self, x):
        x = tf.unstack(x, axis=-1)        
        x = tf.stack([func(x_) for x_, func in zip(x, self.funcs)],axis=-1)
        return x
        
    def adapt(self, data): # deprecated, use MinMax.adapt() instead
        scaled_data = data.map(lambda *x: self.scale(x[0]))
        self.minMax.adapt(scaled_data)

    @tf.function
    def call(self, x):
        return self.scale(x)


# custom binary accuracy metric to handle extra targets        
class MyBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.expand_dims(y_true[...,0], axis=-1)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

# custom accuracy metric to handle extra targets
class MyMSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, log=False, predmin=0, **kwargs):
        super().__init__(**kwargs)
        self.log = log
        self.predmin = predmin

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.expand_dims(y_true[...,0], axis=-1)
        if self.log: y_true = tf.math.log(tf.math.log(y_true - self.predmin + 1) + 1)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

# custom AUC metric to handle extra targets
class MyAUC(tf.keras.metrics.AUC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.expand_dims(y_true[...,0], axis=-1)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

# custom callback for model checkpointing and metric plotting during training
class ModelMonitoring(tf.keras.callbacks.Callback):   
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, epoch, logs=None):

        def ApplyStyle(h, ymin, ymax, color, lstyle=ROOT.kSolid):
            h.SetLineColor(color)
            h.SetLineWidth(2)
            h.SetLineStyle(lstyle)
            if (lstyle != ROOT.kSolid):
                h.SetLineWidth(3)
            h.SetMaximum(ymax)
            h.SetMinimum(ymin)
            h.GetXaxis().SetTitle("# epoch")
            h.GetYaxis().SetTitle("Loss")
            return

        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        last_epoch = len(self.losses)
        
        if (opts.epochs < 50):
            return
        if (last_epoch % (opts.epochs/40) != 0):
            return
    
        plot.CreateDir("%s/modelMonitoring" % opts.saveDir)

        # Save the model
        modelName = "model_%sEpochs.h5" % (last_epoch)
        self.model.save(os.path.join("%s/modelMonitoring" % opts.saveDir,  modelName) )
        
        trainLossList = self.losses
        valLossList   = self.val_losses
        epochList     = range(last_epoch)
        jsonWr = JsonWriter(saveDir=".", verbose=opts.verbose)
        
        ROOT.gStyle.SetOptStat(0)
        canvas = ROOT.TCanvas()
        canvas.cd()
        leg=plot.CreateLegend(0.6, 0.70, 0.9, 0.88)

        h_TrainLoss = ROOT.TGraph(len(epochList), array.array('d', epochList), array.array('d', trainLossList))
        h_ValLoss   = ROOT.TGraph(len(epochList), array.array('d', epochList), array.array('d', valLossList))
        
        ymax = max(h_TrainLoss.GetHistogram().GetMaximum(), h_ValLoss.GetHistogram().GetMaximum())
        ymin = min(h_TrainLoss.GetHistogram().GetMinimum(), h_ValLoss.GetHistogram().GetMinimum())

        ApplyStyle(h_TrainLoss, 0, ymax*1.1,  ROOT.kGreen+1)
        ApplyStyle(h_ValLoss, 0, ymax*1.1,  ROOT.kOrange+3, ROOT.kDashed)
                
        h_TrainLoss.Draw("AC")
        h_ValLoss.Draw("C same")
        leg.AddEntry(h_TrainLoss, "Loss (train)","l")
        leg.AddEntry(h_ValLoss, "Loss (val)","l")

        leg.Draw()
        plot.SavePlot(canvas, "%s/modelMonitoring" % opts.saveDir, "Loss_%sEpochs" % last_epoch)
        canvas.Close()

        return
    
# keras learning rate schedule example, reduceLROnPlateau preferred over this, can be improved
def lr_scheduler(epoch, lr):
    if epoch < 50: # keep initial learning rate for 100 epochs
        return lr
    else:
        return lr * tf.math.exp(-0.025) # decrease exponentially

# https://towardsdatascience.com/custom-loss-function-in-tensorflow-2-0-d8fa35405e4e
# https://github.com/sol0invictus/Blog_stuff/blob/master/custom%20loss/high_level_keras.py
def disco_loss(y_true, y_pred, sample_weight=tf.constant(1.)):
    def flatten_and_cast(x, type):
        return tf.cast(tf.reshape(x, (-1,)), type)
    # Split given labels to the target and the mass value needed for decorrelation
    y_pred = tf.convert_to_tensor(y_pred)
    dcPred = tf.reshape(y_pred, (-1,))
    sample_weights = flatten_and_cast(y_true[:, 2], y_pred.dtype) * sample_weight
    normed_weights = sample_weights * tf.cast(tf.size(sample_weights), tf.float32) / tf.reduce_sum(sample_weights) # ensure weight normalization
    dcMass = flatten_and_cast(y_true[:, 1], y_pred.dtype)
    y_true = flatten_and_cast(y_true[:, 0], y_pred.dtype)
    
    custom_loss = distance_corr(dcMass, dcPred, normedweight=normed_weights, power=1)
    return custom_loss
    
def disco_loss_between_preds(y_true, y_pred, sample_weight=tf.constant(1.)):
    def flatten_and_cast(x, type):
        return tf.cast(tf.reshape(x, (-1,)), type)
    # Split given labels to the target and the mass value needed for decorrelation
    var_1 = y_pred[...,0]
    var_2 = y_pred[...,1]
    dcPred = tf.reshape(var_1, (-1,))
    sample_weights = flatten_and_cast(y_true[:, 2], y_pred.dtype) * sample_weight
    dcMass = tf.reshape(var_2, (-1,))
    
    # The loss
    # crossentropy = tf.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.0)
    custom_loss = distance_corr(dcMass, dcPred, normedweight=sample_weights, power=1)
    return custom_loss
    
def Print(msg, printHeader=False):
    fName = __file__.split("/")[-1]
    if printHeader==True:
        print("=== ", fName)
        print("\t", msg)
    else:
        print("\t", msg)
    return

# wrapper for loss function when y_true contains extra variables
def get_lossfunc_extra_targets(tf_loss):
    def custom_loss(y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true[..., 0], (-1, 1)), y_pred.dtype)
        return tf_loss(y_true, y_pred, sample_weight=sample_weight)
    return custom_loss

# wrapper for custom loss function for the mass regression head
def get_regression_loss(tf_loss, opts, min, max):
    if "squared" in tf_loss.name:
        power = 2
    elif "percentage" in tf_loss.name.lower():
        power = 0
    else: power = 1

    scale = tf.cast(opts.lam_r / tf.math.pow(tf.math.log(tf.math.log(max - opts.masspred_min + 1) + 1), power), tf.float32)

    # the loss function
    def mse_loss(y_true, y_pred):
        log_mass_pred = y_pred
        log_mass_true = tf.expand_dims(tf.math.log(tf.math.log(y_true[...,0] - opts.masspred_min + 1) + 1), axis=-1)
        sample_weight = tf.expand_dims(y_true[...,1], axis=-1)
        return tf_loss(log_mass_true, log_mass_pred, sample_weight=sample_weight) * scale
    return mse_loss

# wrapper for distance correlation metric
def get_disco():

    @tf.function(experimental_relax_shapes=True)
    def DisCo_metric(y_true, y_pred, sample_weights=tf.constant(1.)):
        y_pred = tf.convert_to_tensor(y_pred)
        mass = tf.cast(tf.reshape(y_true[:, 1], (-1, 1)), y_pred.dtype)
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)

        #Type casting and reshaping
        numEntries = tf.cast(tf.size(y_true), dtype=tf.float32)
        numSignalEntries = tf.cast(tf.reduce_sum(y_true), dtype=tf.float32)
        bitMaskForBkg = 1 - y_true

        #Calculate the weights only for background, while setting signal weights to zero (we're only
        #interested in decorrelating the background from mass, signal doesn't matter)
        #Note that distance_corr function expects the weights to be normalized to the number of events
        weightFactor = sample_weights * numEntries / (numEntries - numSignalEntries)
        weights = tf.multiply(weightFactor, bitMaskForBkg)

        dcPred = tf.reshape(y_pred, [tf.size(y_pred)])
        dcMass = tf.reshape(mass, [tf.size(mass)])
        weights = tf.cast(tf.reshape(weights, [tf.size(weights)]), y_pred.dtype)

        # return distance_corr(dcMass, dcPred, normedweight=weights, power=1)
        return distance_corr(dcMass, dcPred, normedweight=weights, power=1)

    return DisCo_metric

def PrintFlushed(msg, printHeader=True):
    '''
    Useful when printing progress in a loop
    '''
    msg = "\r\t" + msg
    ERASE_LINE = '\x1b[2K'
    if printHeader:
        print("=== aux.py")
    sys.stdout.write(ERASE_LINE)
    sys.stdout.write(msg)
    sys.stdout.flush()
    return

def Verbose(msg, printHeader=True, verbose=False):
    if not opts.verbose:
        return
    Print(msg, printHeader)
    return

def PrintXYWTestTrain(x_tr, x_val, x_te, y_tr, y_val, y_te, w_tr, w_val, w_te, nEntries, verbose=False):
    if not verbose:
        return
    table = []
    align = "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    table.append("="*110)
    title = align.format("index", "X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test", "W_train", "W_val", "W_test")
    table.append(title)
    table.append("="*110)
    # For-loop: All entries
    #for i,x in enumerate(x_val, 0):
    n = min(len(x_tr), len(x_val), len(x_te))
    for i in range(n):
        msg = align.format("%d" % i, "%.2f" %  x_tr[i], "%.2f" %  x_val[i], "%.2f" %  x_te[i], 
                           "%.2f" %  y_tr[i], "%.2f" %  y_val[i], "%.2f" %  y_te[i], 
                           "%.2f" % w_tr[i], "%.2f" %  w_val[i], "%.2f" % w_te[i])
        if i < (nEntries/2):
            table.append(ss + msg + ns)
        else:
            table.append(es + msg + ns)
    table.append("="*100)
    for i,r in enumerate(table, 0):
        Print(r, i==0)
    return

def PrintNetworkSummary(opts):
    table    = []
    msgAlign = "{:^10} {:^10} {:>12} {:>10}"
    title    =  msgAlign.format("Layer #", "Neurons", "Activation", "Type")
    hLine    = "="*len(title)
    table.append(hLine)
    table.append(title)
    table.append(hLine)
    for i, n in enumerate(opts.neurons, 0): 
        layerType = "unknown"
        if i == 0:
            layerType = "hidden"
        elif i+1 == len(opts.neurons):
            layerType = "output"
        else:
            layerType = "hidden"
        table.append( msgAlign.format(i+1, opts.neurons[i], opts.activation[i], layerType) )
    table.append("")

    Print("Will construct a DNN with the following architecture", True)
    for r in table:
        Print(r, False)
    return

def GetModelWeightsAndBiases(nInputs, neuronsList):
    '''
    https://keras.io/models/about-keras-models/

    returns a dictionary containing the configuration of the model. 
    The model can be reinstantiated from its config via:
    model = Model.from_config(config)
    '''
    nParamsT  = 0
    nBiasT    = 0 
    nWeightsT = 0

    for i, n in enumerate(neuronsList, 0):
        nParams = 0
        nBias   = neuronsList[i]
        if i == 0:
            nWeights = nInputs * neuronsList[i]
        else:
            nWeights = neuronsList[i-1] * neuronsList[i]
        nParams += nBias + nWeights

        nParamsT  += nParams
        nWeightsT += nWeights
        nBiasT += nBias
    return nParamsT, nWeightsT, nBiasT
        
def GetModelParams(model):
    '''
    https://stackoverflow.com/questions/45046525/how-can-i-get-the-number-of-trainable-parameters-of-a-model-in-keras
    '''
    # For older keras versions check the link above
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

    total_count = trainable_count + non_trainable_count
    return total_count, trainable_count, non_trainable_count

def GetModelConfiguration(myModel, verbose):
    '''
    NOTE: For some older releases of Keras the
    Model.get_config() method returns a list instead of a dictionary.
    This is fixed in newer releases. See details here:
    https://github.com/keras-team/keras/pull/11133
    '''
    
    config = myModel.get_config()
    for i, c in enumerate(config, 1):
        Verbose( "%d) %s " % (i, c), True)
    return config

def GetKwargs(var, standardise=False):

    kwargs  = {
        "normalizeToOne": True,
        "xTitle" : "DNN output",
        "yTitle" : "a.u.",
        "xMin"   :  0.0,
        "xMax"   : +1.0,
        "nBins"  : 200,
        "log"    : True,
        }

    if "mass" in var.lower():
        kwargs["normalizeToOne"] = True
        kwargs["nBins"]  = 50
        kwargs["xMin"]   =  0.0
        kwargs["xMax"]   =  3100
        kwargs["yMin"]   =  0.01
        kwargs["yMax"]   = 1.
        kwargs["xTitle"] = "DNN mass prediction"
        kwargs["yTitle"] = "a.u."
        kwargs["log"]    = True
        kwargs["xBins"]  = datasetUtils.MT_BINS
        if var.lower().split("_")[0] == "output":
            kwargs["legHeader"]  = "all data"
        if var.lower().split("_")[0] == "outputpred":
            kwargs["legHeader"]  = "all data"
            kwargs["legEntries"] = ["train", "test"]
        if var.lower().split("_")[0] == "outputtrain":
            kwargs["legHeader"]  = "training data"
        if var.lower().split("_")[0] == "outputtest":
            kwargs["legHeader"]  = "test data"
        return kwargs

    if "output" in var.lower() or var.lower() == "output":
        kwargs["normalizeToOne"] = True
        kwargs["nBins"]  = 50
        kwargs["xMin"]   =  0.0
        kwargs["xMax"]   =  1.0
        kwargs["yMin"]   =  1e-5
        kwargs["yMax"]   =  2.0
        kwargs["xTitle"] = "DNN output"
        kwargs["yTitle"] = "a.u."
        kwargs["log"]    = True
        if var.lower() == "output":
            kwargs["legHeader"]  = "all data"
        if var.lower() == "outputpred":
            kwargs["legHeader"]  = "all data"
            kwargs["legEntries"] = ["train", "test"]
        if var.lower() == "outputtrain":
            kwargs["legHeader"]  = "training data"
        if var.lower() == "outputtest":
            kwargs["legHeader"]  = "test data"
        return kwargs

    if "phi" in var.lower():
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   = 0.0
        kwargs["xMax"]   = 1.0
        kwargs["nBins"]  = 200
        kwargs["xTitle"] = "DNN output"
        kwargs["yTitle"] = "efficiency" # (\varepsilon)"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if "efficiency" in var:
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   = 0.0
        kwargs["xMax"]   = 1.0
        kwargs["nBins"]  = 200
        kwargs["xTitle"] = "DNN output"
        kwargs["yTitle"] = "efficiency" # (\varepsilon)"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if "roc" in var:
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   = 0.0
        kwargs["xMax"]   = 1.0
        kwargs["nBins"]  = 200
        kwargs["xTitle"] = "signal efficiency"
        kwargs["yTitle"] = "background efficiency"
        kwargs["yMin"]   = 1e-4
        kwargs["yMax"]   = 10
        kwargs["log"]    = True

    if var == "significance":
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   = 0.0
        kwargs["xMax"]   = 1.0
        kwargs["nBins"]  = 200
        kwargs["xTitle"] = "DNN output"
        kwargs["yTitle"] = "significance"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if var == "loss":
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   =  0.
        kwargs["xMax"]   = opts.fold_epochs
        kwargs["nBins"]  = opts.fold_epochs
        kwargs["xTitle"] = "epoch"
        kwargs["yTitle"] = "loss"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if var == "acc":
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   =  0.0
        kwargs["xMax"]   = opts.fold_epochs
        kwargs["nBins"]  = opts.fold_epochs
        kwargs["xTitle"] = "epoch"
        kwargs["yTitle"] = "accuracy"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if var == "auc":
        kwargs["normalizeToOne"] = False
        kwargs["xMin"]   =  0.0
        kwargs["xMax"]   = opts.fold_epochs
        kwargs["nBins"]  = opts.fold_epochs
        kwargs["xTitle"] = "epoch"
        kwargs["yTitle"] = "AUC"
        kwargs["yMin"]   = 0.0
        kwargs["log"]    = False

    if standardise:
        kwargs["xMin"]  =  -5.0
        kwargs["xMax"]  =  +5.0
        kwargs["nBins"] = 500 #1000

    return kwargs

def GetTime(tStart):
    tFinish = time.time()
    dt      = int(tFinish) - int(tStart)
    days    = divmod(dt,86400)      # days
    hours   = divmod(days[1],3600)  # hours
    mins    = divmod(hours[1],60)   # minutes
    secs    = mins[1]               # seconds
    return days, hours, mins, secs

def SaveModelParameters(myModel, nInputs):
    nParams = myModel.count_params()
    total_count, trainable_count, non_trainable_count  = GetModelParams(myModel)
    nParams, nWeights, nBias = GetModelWeightsAndBiases(nInputs, opts.neurons)
    opts.modelParams = total_count
    opts.modelParamsTrain = trainable_count
    opts.modelParamsNonTrainable = non_trainable_count
    opts.modelWeights = nWeights
    opts.modelBiases  = nBias

    ind  = "{:<6} {:<30}"
    msg  = "The model has a total of %s%d parameters%s (neuron weights and biases):" % (hs, opts.modelParams, ns)
    msg += "\n\t" + ind.format("%d" % opts.modelWeights, "Weights")
    msg += "\n\t" + ind.format("%d" % opts.modelBiases, "Biases")
    msg += "\n\t" + ind.format("%d" % opts.modelParamsTrain, "Trainable")
    msg += "\n\t" + ind.format("%d" % opts.modelParamsNonTrainable, "Non-trainable")
    Print(msg, True)
    return

def writeCfgFile(opts):
    # Write to json file
    jsonWr = JsonWriter(saveDir=opts.saveDir, verbose=opts.verbose)
    jsonWr.addParameter("keras version", opts.keras)
    jsonWr.addParameter("host name", opts.hostname)
    jsonWr.addParameter("python version", opts.python)
    jsonWr.addParameter("model", "disCo")
    jsonWr.addParameter("model parameters (total)", opts.modelParams)
    jsonWr.addParameter("model parameters (trainable)", opts.modelParamsTrain)
    jsonWr.addParameter("model parameters (non-trainable)", opts.modelParamsNonTrainable)
    jsonWr.addParameter("model weights", opts.modelWeights)
    jsonWr.addParameter("model biases", opts.modelBiases)
    jsonWr.addParameter("planing", opts.planing)
    jsonWr.addParameter("comput disco only with bkg", opts.onlyBackground)
    jsonWr.addParameter("rndSeed", opts.rndSeed)
    jsonWr.addParameter("layers", len(opts.neurons))
    jsonWr.addParameter("hidden layers", len(opts.neurons)-2)
    jsonWr.addParameter("activation functions", [a for a in opts.activation])
    jsonWr.addParameter("neurons", [n for n in opts.neurons])
    jsonWr.addParameter("loss function", opts.lossFunction)
    jsonWr.addParameter("optimizer", opts.optimizer)
    jsonWr.addParameter("epochs", opts.epochs)
    jsonWr.addParameter("batch size", opts.batchSize)
    jsonWr.addParameter("lambda_d", opts.lam_d)
    jsonWr.addParameter("lambda_r", opts.lam_r)    
    jsonWr.addParameter("elapsed time", opts.elapsedTime)
    for i,b in enumerate(opts.inputList, 1):
        jsonWr.addParameter("var%d"% i, b)
    jsonWr.write(opts.cfgJSON)
    return

def writeGitFile(opts):
    # Write to json file
    path  = os.path.join(opts.saveDir, "gitBranch.txt")
    gFile = open(path, 'w')
    gFile.write(str(opts.gitBranch))

    path  = os.path.join(opts.saveDir, "gitStatus.txt")
    gFile = open(path, 'w')
    gFile.write(str(opts.gitStatus))

    path  = os.path.join(opts.saveDir, "gitDiff.txt")
    gFile = open(path, 'w')
    gFile.write(str(opts.gitDiff))
    return

def PrintXYW(X, Y, W, W_disCo, nEntries, verbose=False):
    if not verbose:
        return

    x  = X[0:nEntries, opts.input_feature_keys.index("mt")].tolist()
    y  = Y[0:nEntries, 0].tolist()
    w  = W[0:nEntries].tolist()
    wd = W_disCo[0:nEntries].tolist()

    table = []
    align = "{:>10} {:>10} {:>10} {:>10} {:>10}"
    table.append("="*100)
    title = align.format("index", "X", "Y", "W", "W DisCo")
    table.append(title)
    table.append("="*100)
    # For-loop: All entries
    for i,xx in enumerate(x, 0):
        msg = align.format("%d" % i, "%.2f" %  x[i], "%.2f" %  y[i], "%.2f" %  w[i], "%.2f" %  wd[i])
        if i < (nEntries/2):
            table.append(ss + msg + ns)
        else:
            table.append(es + msg + ns)
    table.append("="*100)
    for i,r in enumerate(table, 0):
        Print(r, i==0)
    return

def add_extra_targets(dataset, opts):
    """
    This function takes care of dataset threading options and
    transforming the dataset to fit the given nn algorithm
    (parametrization, disCo loss and/or mass regression)
    """

    if opts.mass_index == len(opts.input_feature_keys):
        inputmap = lambda X: X[...,:-1]
    elif opts.mass_index == 0:
        inputmap = lambda X: X[...,1:]
    else: inputmap = lambda X: tf.concat((X[...,:opts.mass_index], X[...,opts.mass_index + 1:]), axis=-1)

    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = 16
    options.threading.private_threadpool_size = 16
    
    dataset = dataset.map(lambda X,Y,W: (X, tf.cast(Y,tf.float32),W)).with_options(options)


    # add variable to decorrelate with to 
    if opts.disco_idx is not None:
        dataset = dataset.map(lambda X,Y,W: (X, tf.stack((Y, X[opts.disco_idx]), axis = -1),W))
    else:
        dataset = dataset.map(lambda X,Y,W: (X, tf.stack((Y,tf.zeros_like(Y)), axis=-1),W))

    # add sample weight to classification targets
    if opts.onlyBackground:
        dataset = dataset.map(lambda X,Y,W: (X, tf.concat((Y, [(1-Y[0]) * 2]), axis = -1),W))
    else:
        dataset = dataset.map(lambda X,Y,W: (X, tf.concat((Y, [tf.ones_like(W)]), axis = -1),W))

    # save the mass parameter as an extra output for plotting even if it's not used in training
    if not opts.param_mass and not opts.regress_mass:
        return dataset.map(lambda X,Y,W: (inputmap(X), {'class_out': tf.concat((Y, X[opts.mass_index:opts.mass_index+1]),axis=-1)}, W))
    elif opts.param_mass:
        return dataset.map(lambda X,Y,W: (X, {"class_out": Y}, W))
    elif opts.regress_mass: # add true mass as targets and set bg weights for regression to 0
        dataset = dataset.map(
            lambda X,Y,W: (inputmap(X),
                           {"class_out": Y, "mass_out": tf.stack((X[opts.mass_index], Y[0] * 2), axis=-1), "combined": Y},
                           W)
        )
        return dataset

# split a dataset into training and validation portions
def validation_split(dataset, split, n_elems):
    n_train = tf.cast((1 - split) * tf.cast(n_elems, tf.float32), tf.int64)
    train_ds = dataset.take(n_train)
    val_ds = dataset.skip(n_train)
    return train_ds, val_ds

# prepare dataset for iteration: caching, shuffling, batching
def finalize(ds, opts, cachepath, testset=False): #NOTE: WIP testing of dataset shuffling
    ds = ds.batch(opts.batchSize)
    if testset:
        fridx = opts.input_feature_keys.index("mt")
        toidx = opts.mass_index

        def save_mtrue_aslast(x, y, w):
            return x, y, w, x[...,toidx]

        def copy_mt2mtrue(x, y, w, m):
            update = x[...,fridx:fridx+1]
            x = tf.concat([x[...,:toidx], update, x[...,toidx+1:]], axis=-1)
            return x, y, w, m

        ds = ds.map(save_mtrue_aslast).map(copy_mt2mtrue)
    if cachepath is not None:
        ds = ds.cache(cachepath)

    ds = ds.unbatch()
    ds = ds.shuffle(3*opts.batchSize, seed = opts.rndSeed)
    ds = ds.batch(opts.batchSize)
    return ds.prefetch(tf.data.AUTOTUNE)

# combine 2 datasets (signal and background) by interleaving events from each one in turn.
# Optionally transfer parametrized mass from signal events to background
def interleave_2(datasets, opts):
    n = 2
    assert len(datasets) == n
    choice_ds = tf.data.Dataset.range(n).repeat()
    if (not opts.copy_mass) or (not opts.param_mass):
        return tf.data.Dataset.choose_from_datasets(datasets, choice_ds, stop_on_empty_dataset=True)
    else:
        fridx = (0, opts.mass_index)
        toidx = (1, opts.mass_index)
        def copy_m_1to2(x, y, w):
            update = x[fridx] - x[toidx]
            x = tf.tensor_scatter_nd_add(x, [toidx], [update])
            return x, y, w

        interleaved = tf.data.Dataset.choose_from_datasets(datasets, choice_ds, stop_on_empty_dataset=True)
        return interleaved.batch(2, drop_remainder=True).map(copy_m_1to2).unbatch()

def count_elems_list(datasets):
    counts = []
    for ds in datasets:
        counts.append(datasetUtils.get_ds_len(ds))
    return counts

# function to reduce max and min of variables from a dataset
def accumulate_mass_max_min(dataset, opts):
    curr_min = tf.float32.max
    curr_max = tf.float32.min
    for elem in dataset:
        new_min = tf.math.reduce_min(elem[1]["mass_out"][...,0])
        new_max = tf.math.reduce_max(elem[1]["mass_out"][...,0])
        curr_min = tf.math.minimum(curr_min, new_min)
        curr_max = tf.math.maximum(curr_max, new_max)
    return curr_max.numpy(), curr_min.numpy()

def main(opts): 

    # Save start time (epoch seconds)
    tStart = time.time()
    Verbose("Started @ " + str(tStart), True)

    # Do not display canvases & disable screen info
    ROOT.gROOT.SetBatch(ROOT.kTRUE)
    ROOT.gROOT.ProcessLine( "gErrorIgnoreLevel = 1001;")

    # Setup the style
    style = tdrstyle.TDRStyle() 
    style.setOptStat(False) 
    style.setGridX(opts.gridX)
    style.setGridY(opts.gridY)


    # Setting the seed for numpy-generated random numbers
    np.random.seed(opts.rndSeed)
    # Setting the seed for python random numbers
    rn.seed(opts.rndSeed)
    # Setting tensorflow random seed
    tf.random.set_seed(opts.rndSeed)

    # For future use
    jsonWr = JsonWriter(saveDir=opts.saveDir, verbose=opts.verbose)

    # set default batch size
    if opts.batchSize == None:
        opts.batchSize = 1024

    # load transverse mass histograms and sample sizes
    if opts.planing:
        sig_mt_dists, bkg_mt_dist, bg_counts = datasetUtils.load_hists(opts.datadir, opts.k, energy=opts.energy)
    else:
        _, _, bg_counts = datasetUtils.load_hists(opts.datadir, opts.k, energy = opts.energy)
        sig_mt_dists=None
        bkg_mt_dist=None
        
    # load training data
    if opts.disco_idx is not None: mt_idx = opts.disco_idx
    else: mt_idx = 0 # placeholder
    datasets_train = datasetUtils.load(opts.datadir,
                                       opts.k,
                                       bg_counts,
                                       sig_hists = sig_mt_dists,
                                       bkg_hists = bkg_mt_dist,
                                       mt_idx=mt_idx,
                                       marginalized=opts.marginalized_planing,
                                       energy = opts.energy,
                                       exclude = opts.excluded_ds
                                       )  

    sig_datasets_train = [sig for sig, bg in datasets_train]
    sig_datasets_train = [add_extra_targets(ds, opts) for ds in sig_datasets_train]    
    bg_datasets_train = [bg for sig, bg in datasets_train]
    bg_datasets_train = [add_extra_targets(ds, opts) for ds in bg_datasets_train]

    datasets_train = [interleave_2(datasets, opts) for datasets in datasets_train]
    datasets_train = [add_extra_targets(ds, opts) for ds in datasets_train]

    # load test data
    datasets_test = datasetUtils.load(opts.datadir,
                                      opts.k,
                                      bg_counts,
                                      testset=True,
                                      mt_idx = mt_idx,
                                      energy = opts.energy,
                                      exclude = opts.excluded_ds
                                      )

    sig_datasets_test = [sig for sig, bg in datasets_test]
    sig_datasets_test = [add_extra_targets(ds, opts) for ds in sig_datasets_test]    
    bg_datasets_test = [bg for sig, bg in datasets_test]
    bg_datasets_test = [add_extra_targets(ds, opts) for ds in bg_datasets_test]

    datasets_test = [interleave_2(datasets, opts) for datasets in datasets_test]
    datasets_test = list(map(lambda ds: add_extra_targets(ds, opts), datasets_test))


    nInputs = datasets_train[0].element_spec[0].shape[-1]

    # get datasets separated to signal, background for later plotting
    sig_datasets_all = list(
        map(
            lambda datasets: datasets[0].concatenate(datasets[1]).filter(lambda *X: X[1]["class_out"][0] == 1),
            zip(datasets_train, datasets_test)
        )
    )

    bg_datasets_all = list(
        map(
            lambda datasets: datasets[0].concatenate(datasets[1]).filter(lambda *X: X[1]["class_out"][0] == 0),
            zip(datasets_train, datasets_test)
        )
    )

    # get dataset sizes for train/validation split
    if not os.path.exists(opts.saveDir + "/train_cache"):
        n_train_plus_val = count_elems_list(datasets_train)
        print(f"total number of elements in training + validation datasets in different folds:\n{n_train_plus_val}")
    else: # datasets will be loaded from cache so there is no need to count elements
        n_train_plus_val = 100000

    # perform train-validation split for each fold
    datasets_train, datasets_val = zip(
        *map(
            lambda x: validation_split(x[0], opts.val_split, x[1]),
            zip(datasets_train, n_train_plus_val)
        )
    )

    # decide whether to train all folds or a specific one
    if opts.worker_id == None:
        folds_to_train = range(opts.k)                               # train all k models
    else: folds_to_train = range(opts.worker_id, opts.worker_id + 1) # train only the model corresponding to worker_id


    opts.parentDir = opts.saveDir
    for n_fold in folds_to_train:
        opts.saveDir = opts.parentDir + f"/fold_{n_fold}/" # save results of each model into their own directories
        
        # make subdirectories if they don't already exist
        if not os.path.exists(opts.saveDir):
            os.mkdir(opts.saveDir)
        if not os.path.exists(opts.saveDir + "train_cache"):
            os.mkdir(opts.saveDir + "train_cache")
        if not os.path.exists(opts.saveDir + "val_cache"):
            os.mkdir(opts.saveDir +"val_cache")
        if not os.path.exists(opts.saveDir + "test_cache"):
            os.mkdir(opts.saveDir +"test_cache")

        # finalize datasets
        cachedirs = [opts.saveDir + var + "_cache/" if opts.use_cache else None for var in ["train", "val", "test"]]
        fold_ds = finalize(datasets_train[n_fold], opts, cachedirs[0])
        fold_val_ds = finalize(datasets_val[n_fold], opts, cachedirs[1])
        fold_test_ds = finalize(datasets_test[n_fold], opts, cachedirs[2], testset = opts.param_mass)

        Verbose("Creating the Keras model", True, opts.verbose)

        # define input layer
        inputs = tf.keras.Input(shape = fold_ds.element_spec[0].shape[1:])
        # layer to drop undesired input variables
        in_sele = InputSelector([var != "_" for var in opts.input_feature_keys])
        x = in_sele(inputs)
        kept_inputs = [var for var in opts.input_feature_keys if var != "_"]

        # layer to transform each variable with their own specific functions
        sanitizer = Sanitizer(kept_inputs)
        x = sanitizer(x)

        # optional input feature standardization
        if opts.standardize:
            stdzr = MinMaxScaler()
            Print("Adding input standardizer layer")
            stdzr.adapt(fold_ds.map(lambda *x: sanitizer(in_sele(x[0])))) # adapt the preprocessing layer to the inputs
            x = stdzr(x)

        # calculate the scale and location of the targets so that the nn outputs can be
        # scaled up s.t. an activation of 1 maps to the largest mass in the dataset.
        if opts.regress_mass:
            mass_max, mass_min = accumulate_mass_max_min(fold_ds, opts)
            pred_max = np.log(np.log(mass_max + 1) + 1).astype(np.float32)

        # loop over hidden layers
        final_layer = len(opts.neurons) - 1
        for iLayer, n in enumerate(opts.neurons, 0):
            layer = "layer#%d" % (int(iLayer)+1)
            if iLayer == len(opts.neurons)-1:
                layer += " (output Layer)" # Layers of nodes between the input and output layers. There may be one or more of these layers.
            else:            
                layer += " (hidden layer)" # A layer of nodes that produce the output variables.
                
            Print("Adding %s, with %s%d neurons%s and activation function %s" % (hs + layer + ns, ls, n, ns, ls + opts.activation[iLayer] + ns), iLayer==0)

            # if regressing mass, add second output to final layer
            if iLayer == final_layer and opts.regress_mass:
                class_pred = Dense(1)(x)
                class_pred = Activation(opts.activation[iLayer], name = "class_out")(class_pred)
                mass_pred = Dense(1)(x)
                mass_pred = Activation(opts.mass_act)(mass_pred)
                mass_pred = tf.keras.layers.Lambda(lambda x: x * pred_max, name = "mass_out")(mass_pred)
                combined_pred = Concatenate(axis=-1, name="combined_out")([class_pred, mass_pred])
                outputs = {'class_out': class_pred, 'mass_out': mass_pred, 'combined': combined_pred}
            # define model output layer
            elif iLayer == final_layer:
                x = Dense(opts.neurons[iLayer])(x)
                outputs = {'class_out': Activation(opts.activation[iLayer], name = "class_out")(x)}
            # define hidden layer
            else:
                x = Dense(opts.neurons[iLayer])(x)
                x = Activation(opts.activation[iLayer])(x)
            if opts.batchnorm[iLayer]:
                x = BatchNormalization()(x)
            if opts.dropout > 0.:
                x = Dropout(opts.dropout, seed=opts.rndSeed)(x)

        # define model by the transformation from inputs to outputs
        myModel = tf.keras.Model(inputs = inputs, outputs = outputs)

        # add losses to model
        if opts.disco_idx is not None and not opts.planing: # add distance correlation term to loss
            Print(f"Adding distance correlation loss to input {opts.disco_varname}", True)
            lossFunc_ = lambda y_true, y_pred, sample_weight=tf.constant(1.): (
                get_lossfunc_extra_targets(opts.tf_loss_func)(y_true, y_pred, sample_weight=sample_weight) 
                + opts.lam_d * disco_loss(y_true, y_pred, sample_weight=sample_weight)
            )
        else:
            lossFunc_ = get_lossfunc_extra_targets(opts.tf_loss_func)
        if opts.regress_mass: # add loss for the mass regression term
                if opts.mass_loss == "MSE":
                    loss_instance = tf.keras.losses.MeanSquaredError()
                elif opts.mass_loss == "MAPE":
                    loss_instance = tf.keras.losses.MeanAbsolutePercentageError()
                else:
                    raise Exception("unknown mass regression loss function " + opts.mass_loss)
                mass_loss = get_regression_loss(loss_instance, opts, mass_min, mass_max)
                corr_loss = lambda y_true, y_pred, sample_weight=tf.constant(1.): opts.lam_d * disco_loss_between_preds(y_true, y_pred, sample_weight=sample_weight)
                lossFunc = {'class_out': lossFunc_, 'mass_out': mass_loss, 'combined': corr_loss}
        else:
            lossFunc = {'class_out': lossFunc_}

        # add metrics
        metrics = {
            'class_out': [
                MyBinaryAccuracy(),
                MyAUC(name="roc_AUC"),
                # get_lossfunc_extra_targets(opts.tf_loss_func)
            ]
        }
        if opts.disco_idx is not None:
            metrics['class_out'].append(get_disco())

        if opts.regress_mass:
            metrics['mass_out'] = [
                MyMSE(log=True),
                # get_regression_loss(mse_instance, opts, mass_min, mass_max)
            ]
            metrics['combined'] = [
                # lambda y_true, y_pred, sample_weight=tf.constant(1.): opts.lam_d * disco_loss_between_preds(y_true, y_pred, sample_weight=sample_weight)
            ]

        # Compile the model
        Print("Compiling the model with the loss function %s and optimizer %s " % (ls + opts.lossFunction + ns, ls + opts.optimizer + ns), True)
        optimizer_kwargs = {"learning_rate": opts.lr}
        if opts.optimizer == 'adamw':
            optimizer_kwargs['weight_decay'] = opts.wd
        optimizer_instance = opts.optimizer_class(**optimizer_kwargs)
        myModel.compile(optimizer=optimizer_instance, loss=lossFunc,  weighted_metrics=metrics)

        # Print a summary representation of your model
        if True: # opts.verbose:
            Print("Printing model summary:", True)
            myModel.summary()

        # Get the number of parameters of the model
        SaveModelParameters(myModel, nInputs) # Compatible with tensorflow v2

        # Get a dictionary containing the configuration of the model. 
        model_cfg = GetModelConfiguration(myModel, opts.verbose)
            
        # Serialize model to JSON (contains arcitecture of model)
        model_json = myModel.to_json() 
        with open(opts.saveDir + "/model_architecture.json", "w") as json_file:
            json_file.write(model_json)

        # Callbacks (https://keras.io/callbacks/)
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=opts.earlystop_patience)
        plot.CreateDir("%s/checkPoint" % opts.saveDir)
        checkPoint = tf.keras.callbacks.ModelCheckpoint("%s/checkPoint/weights.{epoch:02d}-{val_loss:.2f}.h5" % opts.saveDir, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1) #pediod -> save_freq when using tf.keras.callbacks
        backup = tf.keras.callbacks.BackupAndRestore("%s/backup" % opts.saveDir)
        if not opts.reduce_on_plateau:
            lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        elif opts.optimizer == "adamw":
            lr_callback = myReduceLRWDOnPlateau(monitor='val_loss', patience = opts.patience, factor = opts.lr_fac)
        else:
            lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience = opts.patience, factor=opts.lr_fac)
        callbacks = [earlystop, ModelMonitoring(), checkPoint, backup, lr_callback]


        if opts.load_model:
            seqModel = myModel.load_weights(opts.saveDir + "model_weights.h5")
        else:
            # Fit the model with our data
            seqModel = myModel.fit(fold_ds,
                                validation_data = fold_val_ds,
                                epochs          = opts.epochs,    # a full pass over all of your training data
                                shuffle         = False,
                                verbose         = 1,              # 0=silent, 1=progress, 2=mention the number of epoch
                                callbacks       = callbacks,
                                )
            
        # Retrieve  the training / validation loss / accuracy at each epoch
        if not opts.load_model:
            Verbose("The available history objects of the model are: %s" % (", ".join(seqModel.history.keys())), True, opts.verbose)
            epochList = range(opts.epochs)


        out_keys = fold_ds.element_spec[1].keys()

        # get numpy arrays of some of the datasets, possibly truncated to opts.max_elems
        X_sig   = next(iter(sig_datasets_all[n_fold].batch(opts.max_elems)))[0].numpy().reshape((-1,nInputs))
        X_bkg   = next(iter(bg_datasets_all[n_fold].batch(opts.max_elems)))[0].numpy().reshape((-1,nInputs))
        XYW_train = next(iter(fold_ds.unbatch().batch(opts.max_elems)))
        XYW_test = next(iter(fold_test_ds.unbatch().batch(opts.max_elems)))
        X_train = XYW_train[0].numpy().reshape((-1,nInputs))
        X_test  = XYW_test[0].numpy().reshape((-1,nInputs))
        Y_train = {}
        Y_test  = {}
        for key in out_keys:
            n_vals = fold_ds.element_spec[1][key].shape[-1]
            Y_train[key] = XYW_train[1][key].numpy().reshape((-1,n_vals))
            Y_test[key]  = XYW_test[1][key].numpy().reshape((-1,n_vals))
        W_sig = next(iter(sig_datasets_all[n_fold].batch(opts.max_elems)))[2].numpy().reshape((-1,1))
        W_bkg = next(iter(bg_datasets_all[n_fold].batch(opts.max_elems)))[2].numpy().reshape((-1,1))
        W_train = XYW_train[2].numpy().reshape((-1,1))
        W_test = XYW_test[2].numpy().reshape((-1,1))


        if opts.plotInputs:
            std_ = False # (opts.standardise != None)
            Verbose("Plotting all %d input variables for signal and bacgkround" % (nInputs), True)
            for i, var in enumerate(opts.input_feature_keys, 0):
                
                if var == "_": continue

                # Get the lists
                sigList   = X_sig[:,i].tolist()
                bkgList   = X_bkg[:,i].tolist()
                trainList = X_train[:,i].tolist()
                testList  = X_test[:,i].tolist()

                # Make the plots
                func.PlotInputs(sigList  , bkgList , f"{i}_{var}", "%s/%s" % (opts.saveDir, "sigVbkg")   , opts.saveFormats, pType="sigVbkg"   , standardise=std_, w1=W_sig, w2 = W_bkg)
                func.PlotInputs(trainList, testList, f"{i}_{var}", "%s/%s" % (opts.saveDir, "trainVtest"), opts.saveFormats, pType="trainVtest", standardise=std_, w1 = W_train, w2 = W_test)

        # Write the model
        if not opts.load_model:
            modelName = "model_trained.h5"
                
            # Serialize weights to HDF5
            myModel.save_weights(os.path.join(opts.saveDir, 'model_weights.h5'), overwrite=True)
            myModel.save(os.path.join(opts.saveDir, modelName))
                
            # Write weights and architecture in txt file
            scaler_attributes = "scalerType None\n"
            modelFilename = os.path.join(opts.saveDir, "model.txt")
            Print("Writing the model (weights and architecture) in the file %s" % (hs + os.path.basename(modelFilename) + ns), True)
            func.WriteModel(myModel, model_json, opts.inputList, scaler_attributes, modelFilename, verbose=False)

        # Produce method score (i.e. predict output value for given input dataset). Computation is done in batches.
        Print("Generating output predictions (numpy arrays) for the input samples", True)
        pred_train  = myModel.predict(X_train, batch_size=None, verbose=1, steps=None) # DNN output for training data (for both signal & bkg)
        pred_test   = myModel.predict(X_test , batch_size=None, verbose=1, steps=None) # DNN output for test data (for both signal & bkg)
        pred_signal = myModel.predict(X_sig  , batch_size=None, verbose=1, steps=None) # DNN output for signal only (all data)
        pred_bkg    = myModel.predict(X_bkg  , batch_size=None, verbose=1, steps=None) # DNN output for data only (all data)


        XYW_train = np.concatenate((X_train, Y_train["class_out"], W_train), axis=1)
        XYW_test  = np.concatenate((X_test , Y_test[ "class_out"], W_test), axis=1)

        # Pick events with output = 1
    #    Verbose("Select events/samples which have an output variable Y (last column) equal to 1 (i.e. prediction is combatible with signal)", True)
        X_train_S = XYW_train[XYW_train[:,nInputs] == 1]; X_train_S = X_train_S[:,0:nInputs]
        X_test_S  = XYW_test[XYW_test[:,nInputs] == 1];   X_test_S  = X_test_S[:,0:nInputs]

        w_train_S = XYW_train[XYW_train[:,nInputs] == 1]; w_train_S  = w_train_S[:,-1]
        w_test_S  = XYW_test[XYW_test[:,nInputs] == 1];   w_test_S = w_test_S[:,-1]

    #    Verbose("Select events/samples which have an output variable Y (last column) equal to 0 (i.e. prediction is NOT combatible with signal)", False)
        X_train_B = XYW_train[XYW_train[:,nInputs] == 0]; X_train_B = X_train_B[:,0:nInputs]
        X_test_B  = XYW_test[XYW_test[:,nInputs] == 0];   X_test_B  = X_test_B[:,0:nInputs]
        
        w_train_B = XYW_train[XYW_train[:,nInputs] == 0]; w_train_B  = w_train_B[:,-1]
        w_test_B  = XYW_test[XYW_test[:,nInputs] == 0];   w_test_B = w_test_B[:,-1]

        if opts.param_mass:
            mtrue_test = next(iter(fold_test_ds.unbatch().batch(opts.max_elems)))[3].numpy().reshape((-1,1))
            mtrue_test_S = mtrue_test[Y_test["class_out"][...,0] == 1]
        elif opts.regress_mass:
            mtrue_test_S = Y_test['mass_out'][XYW_test[:,nInputs] == 1]; mtrue_test_S = mtrue_test_S[:,0]
        else:
            mtrue_test_S = Y_test['class_out'][XYW_test[:,nInputs] == 1]; mtrue_test_S = mtrue_test_S[...,-1]

        # Produce method score for signal (training and test) and background (training and test)
        pred_train_S =  myModel.predict(X_train_S, batch_size=None, verbose=1, steps=None)
        pred_train_B =  myModel.predict(X_train_B, batch_size=None, verbose=1, steps=None)
        pred_test_S  =  myModel.predict(X_test_S , batch_size=None, verbose=1, steps=None)
        pred_test_B  =  myModel.predict(X_test_B , batch_size=None, verbose=1, steps=None)

        # scale regression head outputs to correspond to a mass
        if opts.regress_mass:
            pred_train['mass_out'] = tf.math.exp(tf.math.exp(pred_train['mass_out']) - 1) - 1
            pred_test['mass_out'] = tf.math.exp(tf.math.exp(pred_test['mass_out']) - 1) - 1
            pred_signal['mass_out'] = tf.math.exp(tf.math.exp(pred_signal['mass_out']) - 1) - 1
            pred_bkg['mass_out'] = tf.math.exp(tf.math.exp(pred_bkg['mass_out']) - 1) - 1
            pred_train_S['mass_out'] = tf.math.exp(tf.math.exp(pred_train_S['mass_out']) - 1) - 1
            pred_train_B['mass_out'] = tf.math.exp(tf.math.exp(pred_train_B['mass_out']) - 1) - 1
            pred_test_S['mass_out'] = tf.math.exp(tf.math.exp(pred_test_S['mass_out']) - 1) - 1
            pred_test_B['mass_out'] = tf.math.exp(tf.math.exp(pred_test_B['mass_out']) - 1) - 1

        # Inform user of early stop
        stopEpoch = earlystop.stopped_epoch
        if stopEpoch != 0 and stopEpoch < opts.epochs:
            msg = "Early stop occured after %d epochs!" % (stopEpoch)
            opts.fold_epochs = stopEpoch
            Print(cs + msg + ns, True)
        else:
            opts.fold_epochs = opts.epochs

        # Create json file
        writeGitFile(opts)

        # plot input distributions before and after nn cut
        for i in range(X_test.shape[-1]):
            if opts.input_feature_keys[i] == "_": continue
            func.PlotInputDistortion(pred_test_B['class_out'], X_test_B[...,i], w_test_B, f"bg_{opts.input_feature_keys[i]}{i}_be4vsafter", opts.saveDir + "/b4vsafter/", opts.saveFormats)
            func.PlotInputDistortion(pred_test_S['class_out'], X_test_S[...,i], w_test_S, f"sig_{opts.input_feature_keys[i]}{i}_be4vsafter", opts.saveDir + "/b4vsafter/", opts.saveFormats)

        # Plot selected output and save to JSON file for future use
        for key in out_keys:
            if "combine" in key: continue
            func.PlotAndWriteJSON(pred_signal[key] , pred_bkg[key]    , opts.saveDir, f"Output_{key}"       , jsonWr, opts.saveFormats, **GetKwargs(f"Output_{key}"     )) # DNN score (predicted): Signal Vs Bkg (all data)
            func.PlotAndWriteJSON(pred_train[key]  , pred_test[key]   , opts.saveDir, f"OutputPred_{key}"   , jsonWr, opts.saveFormats, **GetKwargs(f"OutputPred_{key}" )) # DNN score (sig+bkg)  : Train Vs Predict
            func.PlotAndWriteJSON(pred_train_S[key], pred_train_B[key], opts.saveDir, f"OutputTrain_{key}"  , jsonWr, opts.saveFormats, **GetKwargs(f"OutputTrain_{key}")) # DNN score (training) : Sig Vs Bkg
            func.PlotAndWriteJSON(pred_test_S[key] , pred_test_B[key] , opts.saveDir, f"OutputTest_{key}"   , jsonWr, opts.saveFormats, **GetKwargs(f"OutputTest_{key}" )) # DNN score (predicted): Sig Vs Bkg  
        if opts.regress_mass:
            func.PlotAndWriteJSON_DNNscore(pred_test_S['class_out'], pred_test_B['class_out'], 0.5, pred_test_S['mass_out'], pred_test_B['mass_out'], opts.saveDir, 'OutputDist_cut', jsonWr, opts.saveFormats, **GetKwargs(f"mass"))

        rocDict = {}

        # Plot overtraining test
        htrain_s, htest_s, htrain_b, htest_b = func.PlotOvertrainingTest(
            pred_train_S['class_out'],
            pred_test_S['class_out'],
            pred_train_B['class_out'],
            pred_test_B['class_out'],
            opts.saveDir, "OvertrainingTest",
            opts.saveFormats,
            w_train_S,
            w_test_S,
            w_train_B,
            w_test_B
        )

        # Plot summary plot (efficiency & singificance)
        func.PlotEfficiency(htest_s, htest_b, opts.saveDir, "Summary", opts.saveFormats)

        # Write efficiencies (signal and bkg)
        xVals_S, xErrs_S, effVals_S, effErrs_S  = func.GetEfficiency(htest_s)
        xVals_B, xErrs_B, effVals_B, effErrs_B  = func.GetEfficiency(htest_b)
        func.PlotTGraph(xVals_S, xErrs_S, effVals_S, effErrs_S, opts.saveDir, "EfficiencySig", jsonWr, opts.saveFormats, **GetKwargs("efficiency") )
        func.PlotTGraph(xVals_B, xErrs_B, effVals_B, effErrs_B, opts.saveDir, "EfficiencyBkg", jsonWr, opts.saveFormats, **GetKwargs("efficiency") )

        xVals, xErrs, sig_def, sig_alt = func.GetSignificance(htest_s, htest_b)
        func.PlotTGraph(xVals, xErrs, sig_def, effErrs_B, opts.saveDir, "SignificanceA", jsonWr, opts.saveFormats, **GetKwargs("significance") )
        func.PlotTGraph(xVals, xErrs, sig_alt, effErrs_B, opts.saveDir, "SignificanceB", jsonWr, opts.saveFormats, **GetKwargs("significance") )

        # Plot some metrics
        if not opts.load_model:
            xErr = [0.0 for i in range(0, opts.fold_epochs)]
            yErr = [0.0 for i in range(0, opts.fold_epochs)]
            for name, metric in seqModel.history.items():
                if "acc" in name:
                    var = "acc"
                    draw = True
                elif "loss" in name or "DisCo" in name:
                    var = "loss"
                    draw = True
                elif "AUC" in name:
                    var = "auc"
                    draw = True
                else: draw = False
                if draw:
                    func.PlotTGraph(range(opts.fold_epochs), xErr, metric, yErr , opts.saveDir, name, jsonWr, opts.saveFormats, **GetKwargs(var) )

            
        # Plot ROC curve
        gSig  = func.GetROC(htest_s, htest_b)
        gDict = {"graph" : [gSig], "name" : [os.path.basename(opts.saveDir)]}
        func.PlotROC(gDict, opts.saveDir, "ROC", opts.saveFormats)

        if opts.regress_mass:
            func.PlotMassPredictionErrors(mtrue_test_S, pred_test_S['mass_out'], opts.saveDir, "MassPredErrors", jsonWr, opts.saveFormats)

        # plot distribution widths of the variable of interest for each dataset
        if opts.param_mass and opts.disco_idx is not None:
            poi = X_test_S[...,opts.input_feature_keys.index(opts.disco_varname)]
        elif opts.regress_mass:
            poi = pred_test_S['mass_out']
        elif opts.disco_idx is not None:
            poi = X_test_S[...,opts.input_feature_keys.index(opts.disco_varname)]

        if poi is not None:
            plotxMin = -1
            plotxMax = 1
            func.PlotPoivsMassDiffPerDataset(mtrue_test_S, poi, opts.saveDir, "POIvsTrueMass", jsonWr, opts.saveFormats, xMin = plotxMin, xMax = plotxMax)
        func.PlotDNNscorePerDataset(mtrue_test_S, pred_test_S['class_out'], opts.saveDir, "BinnedDNNscores", jsonWr, opts.saveFormats)

        # Write the resultsJSON file!
        jsonWr.write(opts.resultsJSON)
        
        # Print total time elapsed
        days, hours, mins, secs = GetTime(tStart)
        dt = "%s days, %s hours, %s mins, %s secs" % (days[0], hours[0], mins[0], secs)
        Print("Elapsed time: %s" % (hs + dt + ns), True)
        opts.elapsedTime = dt
        writeCfgFile(opts)

        for name in ["/train_cache", "/val_cache", "/test_cache"]:
            shutil.rmtree(opts.saveDir + name)

    return 

#================================================================================================ 
# Main
#================================================================================================ 
# TODO: fix planing/input var selection 
if __name__ == "__main__":
    '''
    https://docs.python.org/3/library/argparse.html
    
    name or flags...: Either a name or a list of option strings, e.g. foo or -f, --foo.
    action..........: The basic type of action to be taken when this argument is encountered at the command line.
    nargs...........: The number of command-line arguments that should be consumed.
    const...........: A constant value required by some action and nargs selections.
    default.........: The value produced if the argument is absent from the command line.
    type............: The type to which the command-line argument should be converted.
    choices.........: A container of the allowable values for the argument.
    required........: Whether or not the command-line option may be omitted (optionals only).
    help............: A brief description of what the argument does.
    metavar.........: A name for the argument in usage messages.
    dest............: The name of the attribute to be added to the object returned by parse_args().
    '''
    
    # Default Settings
    STANDARDIZE  = False
    DATADIR      = "/home/AapoKossi/temp"
#    INPUTVARS    = "tau_pt,pt_miss,R_tau,bjet_pt,tau_mt,mt,dPhiTauNu,dPhitaub,dPhibNu,mtrue"
    INPUTVARS    = "load"
    SAVEDIR      = None
    SAVEDIRAPPDX = False
    SAVEFORMATS  = "pdf"
    URL          = False
    SCALEBACK    = True
    PLANING      = False
    MARGINAL     = False
    ONLYBKG      = False
    DECORRELATE  = None
    VERBOSE      = False
    RNDSEED      = 1234
    EPOCHS       = 100
    BATCHSIZE    = 1024
    ACTIVATION   = "swish,swish,swish,swish,sigmoid" 
    NEURONS      = "1024,512,256,128,1"
    BATCHNORM    = "True,True,True,True,False"
    GRIDX        = False
    GRIDY        = False
    LOSSFUNCTION = 'binary_crossentropy'
    OPTIMIZER    = 'adam'
    CFGJSON      = "config.json"
    RESULTSJSON  = "results.json"
    PLOTINPUTS   = True
    LAMBDAD       = 15
    LAMBDAR       = 4
    FOLDS        = 2
    MAXLOADEDELEMS = 2**18
    DISCOIDX     = None    # by default, NOT using distance correlation loss term
    VALSPLIT     = 0.1
    WORKERID     = None
    ENERGY       = "13"
    LEARNINGRATE = 3e-4
    REGRESSMASS  = False
    MASSPREDMIN  = 0
    PARAMMASS    = False
    EXCLUDE      = "ZZ,WZ,WW,WJets,DYJets"
    REDUCEONPLATEAU = False
    LOADMODEL    = False
    USECACHE     = False
    COPYMASS     = True
    DROPOUT      = 0.15
    WEIGHTDECAY  = 1e-6
    MASSLOSS     = "MSE"
    MASSACTIVATION = "softplus"
    LRFACTOR     = 0.5
    LRPATIENCE   = 10
    EARLYSTOP    = 50
    SHUFFLE      = False

    # TODO: switch to argumentparser, optparser is python 2 legacy
    parser = OptionParser(usage="Usage: %prog [options]")

    parser.add_option("--inputVars", dest="inputVariables", default=INPUTVARS,
                      help="list of names of the model input variables, input _ in order to specify that the input at that index "
                       + "in the loaded dataset should not be used [default: %s]" % INPUTVARS)

    parser.add_option("--decorrelate", dest="decorrelate", default=DECORRELATE,
                      help="Calculate weights to decorrelate a variable from the training. This is done by reweighting the branches so that signal and background have similar mass distributions [default: %s]" % DECORRELATE)

    parser.add_option("--planing", dest="planing", action="store_true", default=PLANING,
                      help="Calculate weights to decorrelate a variable from the training. This is done by reweighting the branches so that the selected variable becomes flat [default: %s]" % PLANING)

    parser.add_option("--marginalized", dest="marginalized_planing", action="store_true", default=MARGINAL,
                      help="If specified planing, whether to plane the marginalized transverse mass distribution. If false and planing, planes all the signal datasets individually. [default: %s]" % MARGINAL)

    parser.add_option("--onlyBackground", dest="onlyBackground", action="store_true", default=ONLYBKG,
                      help="Calculate distance correlation using only the background events (w_sig = 0). [default: %s]" % ONLYBKG)

    parser.add_option("--scaleBack", dest="scaleBack", action="store_true", default=SCALEBACK,
                      help="Scale back the data to the original representation (before the standardisation). i.e. Performing inverse-transform to all variables to get their original representation. [default: %s]" % SCALEBACK)

    parser.add_option("--standardize", dest="standardize", action="store_true", default=STANDARDIZE,
                      help="Whether to insert a sanitizer layer in the beginning of the model. [default: %s]" % STANDARDIZE)

    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=VERBOSE, 
                      help="Enable verbose mode (for debugging purposes mostly) [default: %s]" % VERBOSE)

    parser.add_option("--plotInputs", dest="plotInputs", action="store_true", default=PLOTINPUTS, 
                      help="Enable plotting of input variables [default: %s]" % PLOTINPUTS)

    parser.add_option("--dataDir", dest="datadir", type="string", default=DATADIR, 
                      help="Input directory containing the datasets saved with datasetUtils.Writer) [default: %s]" % DATADIR)

    parser.add_option("--resultsJSON", dest="resultsJSON", default=RESULTSJSON,
                      help="JSON file containing the results [default: %s]" % (RESULTSJSON))

    parser.add_option("--cfgJSON", dest="cfgJSON", default=CFGJSON,
                      help="JSON file containing the configurations [default: %s]" % (CFGJSON))

    parser.add_option("--saveDir", dest="saveDir", type="string", default=SAVEDIR,
                      help="Directory where all pltos will be saved [default: %s]" % SAVEDIR)

    parser.add_option("--url", dest="url", action="store_true", default=URL, 
                      help="Don't print the actual save path the histogram is saved, but print the URL instead [default: %s]" % URL)

    parser.add_option("-s", "--saveFormats", dest="saveFormats", default = SAVEFORMATS,
                      help="Save formats for all plots [default: %s]" % SAVEFORMATS)
    
    parser.add_option("--rndSeed", dest="rndSeed", type=int, default=RNDSEED, 
                      help="Value of random seed (integer) [default: %s]" % RNDSEED)
    
    parser.add_option("--epochs", dest="epochs", type=int, default=EPOCHS, 
                      help="Number of \"epochs\" to be used (how many times you go through your training set) [default: %s]" % EPOCHS)

    parser.add_option("--batchSize", dest="batchSize", type=int, default=BATCHSIZE,
                      help="The \"batch size\" to be used (= a number of samples processed before the model is updated). Batch size impacts learning significantly; typically networks train faster with mini-batches. However, the batch size has a direct impact on the variance of gradients (the larger the batch the better the appoximation and the larger the memory usage). [default: %s]" % BATCHSIZE)

    parser.add_option("--activation", dest="activation", type="string", default=ACTIVATION,
                      help="Type of transfer function that will be used to map the output of one layer to another [default: %s]" % ACTIVATION)

    parser.add_option("--exclude", dest="excluded_ds", type="string", default=EXCLUDE,
                      help="Exclude datasets that include these substrings [default: %s]" % EXCLUDE)

    parser.add_option("--neurons", dest="neurons", type="string", default=NEURONS,
                      help="List of neurons to use for each sequential layer (comma-separated integers)  [default: %s]" % NEURONS)

    parser.add_option("--batchnorm", dest="batchnorm", type="string", default=BATCHNORM,
                      help="List of Booleans to specify whether to insert batch normalization layers after activations (comma-separated python booleans)  [default: %s]" % BATCHNORM)
    
    parser.add_option("--lossFunction", dest="lossFunction", type="string", default=LOSSFUNCTION,
                      help="One of the two parameters required to compile a model. The weights will take on values such that the loss function is minimized [default: %s]" % LOSSFUNCTION)

    parser.add_option("--optimizer", dest="optimizer", type="string", default=OPTIMIZER,
                      help="Name of optimizer function; one of the two parameters required to compile a model: [default: %s]" % OPTIMIZER)

    parser.add_option("--gridX", dest="gridX", action="store_true", default=GRIDX,
                      help="Enable x-axis grid [default: %s]" % GRIDX)

    parser.add_option("--gridY", dest="gridY", action="store_true", default=GRIDY,
                      help="Enable y-axis grid [default: %s]" % GRIDY)

    parser.add_option("--lam_d",dest="lam_d",  default=LAMBDAD,  type = int,   help="Lambda for disCo loss (Default: %s)" % LAMBDAD)

    parser.add_option("--lam_r",dest="lam_r",  default=LAMBDAR,  type = float,   help="Lambda (Default: %s)" % LAMBDAR)

    parser.add_option("--k",dest="k",  default=FOLDS,  type = int,   help="K-folding number of folds used when saving the dataset (Default: %s)" % FOLDS)

    parser.add_option("--maxElems",dest="max_elems",  default=MAXLOADEDELEMS,  type = int,   help="Maximum number of single dataset elements to be loaded in memory at any time, all (Default: %s)" % MAXLOADEDELEMS)

    parser.add_option("--discoIdx",dest="disco_idx",  default=DISCOIDX,  type = int,   help="Index in the training data of the variable to decorrelate NN output with disCo from. Note that this is applied before dropping any input variables (Default: %s)" % DISCOIDX)

    parser.add_option("--valSplit",dest="val_split",  default=VALSPLIT,  type = float,   help="What fraction of the training data to set aside for model validation (Default: %s)" % VALSPLIT)

    parser.add_option("--worker",dest="worker_id",  default=WORKERID,  type = int,   help="Which fold of the data to train the model on, defaults to training all the models sequentially (Default: %s)" % WORKERID)

    parser.add_option("--energy", dest="energy", type="string", default=ENERGY,
                      help="Center of mass energy for the data, used for fetching event cross sections for different bakcgrounds: [default: %s]" % ENERGY)

    parser.add_option("--learning_rate", dest="lr", type=float, default=LEARNINGRATE,
                      help="Initial learning rate for the optimizer [default: %s]" % LEARNINGRATE)

    parser.add_option("--weight_decay", dest="wd", type=float, default=WEIGHTDECAY,
                      help="Initial weight decay rate for the optimizer [default: %s]" % WEIGHTDECAY)

    parser.add_option("--regress_mass", dest="regress_mass", action="store_true", default=REGRESSMASS,
                      help="Specify this to add additional output to the model for regressing the H+ mass[default: %s]" % REGRESSMASS)

    parser.add_option("--mass_loss", dest="mass_loss", type=str, default=MASSLOSS,
                      help="Which loss function to use for regressing the H+ mass[default: %s]" % MASSLOSS)

    parser.add_option("--mass_activation", dest="mass_act", type=str, default=MASSACTIVATION,
                      help="Which activation function to use for regressing the H+ mass [default: %s]" % MASSACTIVATION)

    parser.add_option("--masspred_min", dest="masspred_min", type=int, default=MASSPREDMIN,
                      help="Minimum output for the nn head regressing the H+ mass[default: %s]" % MASSPREDMIN)    

    parser.add_option("--param_mass", dest="param_mass", action="store_true", default=PARAMMASS,
                      help="Specify this to add additional output to the model for parametrizing the model on H+ mass[default: %s]" % PARAMMASS)

    parser.add_option("--reduce_on_plateau", dest="reduce_on_plateau", action="store_true", default=REDUCEONPLATEAU,
                      help="Specify to overwrite lr decay with a keras reduce on plateau callback [default: %s]" % REDUCEONPLATEAU)

    parser.add_option("--load_model", dest="load_model", action="store_true", default=LOADMODEL,
                      help="Load the model from the saveDir [default: %s]" % LOADMODEL)

    parser.add_option("--use_cache", dest="use_cache", action="store_true", default=USECACHE,  help="Use cache for faster dataset loading. Default: %s" % USECACHE)

    parser.add_option("--gen_save_appendix", dest="saveapdx", action="store_true", default=SAVEDIRAPPDX,  help="Add descriptionary appendix to save dir. Default: %s" % SAVEDIRAPPDX)

    parser.add_option("--no_copy_mass", dest="copy_mass", action="store_false", default=COPYMASS,
                      help="Instead of giving background events mass parameter exactly from the previous signal event, take it from a naive predetermined distribution. Default: %s" % COPYMASS)

    parser.add_option("--dropout_rate", dest="dropout", type=float, default=DROPOUT,
                      help="Dropout rate between Dense layers. Default: %s" % DROPOUT)

    parser.add_option("--lr_fac", dest="lr_fac", type=float, default=LRFACTOR,
                      help="Factor to reduce learning rate (and weight decay) by on plateau. Default: %s" % LRFACTOR)

    parser.add_option("--patience", dest="patience", type=float, default=LRPATIENCE,
                      help="number of epochs to keep training before reducing lr on validation loss plateau. Default: %s" % LRPATIENCE)

    parser.add_option("--earlystop", dest="earlystop_patience", type=float, default=EARLYSTOP,
                      help="number of epochs to keep training before early stopping on validation loss plateau. Default: %s" % EARLYSTOP)

    parser.add_option("--shuffle", dest="train_shuffle", action="store_true", default=SHUFFLE,
                      help="shuffle the input training data each epoch [default: %s]" % SHUFFLE)

    (opts, parseArgs) = parser.parse_args()
    
    # Require at least two arguments (script-name, ...)
    if len(sys.argv) < 1:
        parser.print_help()
        sys.exit(1)
    else:
        pass

    # Input list of discriminatin variables
    if opts.inputVariables == "load":
        opts.inputList = datasetUtils.load_input_vars(opts.datadir)
    else:
        opts.inputList = opts.inputVariables.split(",")
    if len(opts.inputList) < 1:
        raise Exception("At least one input variable needed to create the DNN. Only %d provided" % (len(opts.inputList)) )
    multicols = datasetUtils.COLUMN_KEYS.keys()
    opts.input_feature_keys = []
    for variable in opts.inputList:
        if variable not in multicols:
            opts.input_feature_keys.append(variable)
        else:
            [opts.input_feature_keys.append(sub_var) for sub_var in datasetUtils.COLUMN_KEYS[variable]]

    if opts.disco_idx is not None:
        opts.disco_varname = opts.input_feature_keys[opts.disco_idx]

    mass_index = opts.inputList.index("mtrue")
    opts.mass_index = opts.input_feature_keys.index("mtrue")
    if opts.regress_mass or not opts.param_mass:
        opts.inputList.pop(mass_index)
        opts.input_feature_keys.pop(opts.mass_index)

    # Create save formats
    if "," in opts.saveFormats:
        opts.saveFormats = opts.saveFormats.split(",")
    else:
        opts.saveFormats = [opts.saveFormats]
    opts.saveFormats = [s for s in opts.saveFormats]

    # Create specification lists
    if "," in opts.activation:
        opts.activation = opts.activation.split(",")
    else:
        opts.activation = [opts.activation]
    Verbose("Activation = %s" % (opts.activation), True)
    if "," in opts.excluded_ds:
        opts.excluded_ds = opts.excluded_ds.split(",")
    else:
        opts.excluced_ds = [opts.excluded_ds]
    Verbose("Excluded datasets = %s" % (opts.excluded_ds), True)
    if "," in opts.neurons:
        opts.neurons = list(map(int, opts.neurons.split(",")) )
    else:
        opts.neurons = list(map(int, [opts.neurons]))
    Verbose("Neurons = %s" % (opts.neurons), True)
    if "," in opts.batchnorm:
        opts.batchnorm = list(map(lambda val: val == "True", opts.batchnorm.split(",")) )
    else:
        opts.batchnorm = list(map(lambda val: val == "True", [opts.batchnorm] * (len(opts.neurons)-1) + ["False"]))
    Verbose("batch normalization = %s" % (opts.batchnorm), True)
        
    # Sanity checks (One activation function and batchnorm option for each layer)
    if len(opts.neurons) != len(opts.activation) | len(opts.neurons) != len(opts.batchnorm):
        msg = "The list of neurons (size=%d) is not the same size as the list of activation functions (=%d)" % (len(opts.neurons), len(opts.activation))
        raise Exception(es + msg + ns)  
    # Sanity check (Last layer)
    if opts.neurons[-1] != 1:
        if not opts.regress_mass:
            msg = "The number of neurons for the last layer should be equal to 1 (=%d instead)" % (opts.neurons[-1])
            raise Exception(es + msg + ns)
        elif opts.neurons[-1] != 2:
            msg = "The number of neurons for the last layer should be equal to 2 when regressing mass (=%d instead)" % (opts.neurons[-1])
            raise Exception(es + msg + ns)

    # Define dir/logfile names
    specs = "%dLayers" % (len(opts.neurons))
    '''
    for i,n in enumerate(opts.neurons, 0):
        specs+= "_%s%s" % (opts.neurons[i], opts.activation[i])
    '''
    specs+= "_%sEpochs_%sBatchSize" % (opts.epochs, opts.batchSize)    

#    if opts.decorrelate != None:
#        if opts.decorrelate not in opts.inputList:
#            msg = "Cannot apply sample reweighting. The input variable \"%s\" is not in the inputList." % (opts.decorrelate)
#            raise Exception(es + msg + ns)

    # Get the current date and time
    now    = datetime.now()
    nDay   = now.strftime("%d")
    nMonth = now.strftime("%h")
    nYear  = now.strftime("%Y")
    nTime  = now.strftime("%Hh%Mm%Ss") # w/ seconds
    nDate  = "%s%s%s_%s" % (nDay, nMonth, nYear, nTime)
    sName  = "DNNresults_"
#    if opts.decorrelate != None:
#        sName += "_%sDecorrelated" % opts.decorrelate
    if opts.planing:
        sName += "_Planing"
    if opts.marginalized_planing:
        sName += "_Marginalized"
    if opts.onlyBackground:
        sName += "_OnlyBkg"

    # Add the number of input variables
    sName += "_%dInputs" % (len(opts.inputList))
    # Add the lamda value
    if opts.disco_idx is not None or opts.regress_mass:
        sName += "_Lamda_d%s" % (opts.lam_d)
    if opts.regress_mass:
        sName += "_Lamda_r%s" % (opts.lam_r)
    sName += "_lr%s" % (opts.lr)
    if opts.optimizer == "adamw":
        sName += "_wd%s" % (opts.wd)
    # Add the time-stamp last
    sName += "_%s" % (nDate)

    sName += f"_{opts.optimizer}"
    
    if opts.regress_mass:
        sName += "_RegressMass"
    if opts.param_mass:
        sName += "_ParamMass"
    
    # Determine path for saving plots
    if opts.saveDir == None:
        usrName = getpass.getuser()
        usrInit = usrName[0]
        myDir   = ""
        if "lxplus" in socket.gethostname():
            myDir = "/afs/cern.ch/user/%s/%s/public/html/" % (usrInit, usrName)
        else:
            myDir = os.path.join(os.getcwd())
        opts.saveDir = os.path.join(myDir, sName)
    elif opts.saveapdx:
        opts.saveDir = os.path.join(opts.saveDir, sName)

    # Create dir if it does not exist
    if not os.path.exists(opts.saveDir):
        os.makedirs(opts.saveDir)

    # Write list of input variables to savedir for input conversion at inference time
    with open(opts.saveDir + "/input_vars.txt", "w") as f:
        for var in opts.inputList:
            f.write(var + "\n")

    # Inform user of network stup
    PrintNetworkSummary(opts)
#    Print("A total of %s%d input variables%s will be used:\n\t%s%s%s" % (ls, len(opts.inputList),ns, ls, "\n\t".join(opts.inputList), ns), True)

    # See https://keras.io/activations/
    actList = ["elu", "softmax", "selu", "softplus", "softsign", "PReLU", "LeakyReLU",
               "relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear", "swish"] # Loukas used "relu" for resolved top tagger
    # Sanity checks
    for a in opts.activation:
        if a not in actList:
            msg = "Unsupported activation function %s. Please select on of the following:%s\n\t%s" % (opts.activation, ss, "\n\t".join(actList))
            raise Exception(es + msg + ns)    

    if opts.mass_act not in actList:
        msg = "Unsupported activation function %s. Please select on of the following:%s\n\t%s" % (opts.activation, ss, "\n\t".join(actList))
        raise Exception(es + msg + ns)    


    # See https://keras.io/losses/
    lossList = ["binary_crossentropy",
                "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge",
                "hinge", "categorical_hinge", "logcosh", "huber_loss", "categorical_crossentropy", "sparse_categorical_crossentropy", 
                "kullback_leibler_divergenc", "poisson", "cosine_proximity"]
    bLossList = ["binary_crossentropy",]
    # Sanity checks
    if opts.lossFunction not in lossList:
        msg = "Unsupported loss function %s. Please select on of the following:%s\n\t%s" % (opts.lossFunction, ss, "\n\t".join(lossList))
        raise Exception(es + msg + ns)    
    elif opts.lossFunction not in bLossList:
        msg = "Binary output currently only supports the following loss fucntions: %s" % ", ".join(bLossList)
        raise Exception(es + msg + ns)
    
    lossMap = { "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(),
                "mean_squared_error": tf.keras.losses.MeanSquaredError(),
                "mean_absolute_error": tf.keras.losses.MeanAbsoluteError(),
                "mean_absolute_percentage_error": tf.keras.losses.MeanAbsolutePercentageError(),
                "mean_squared_logarithmic_error": tf.keras.losses.MeanSquaredLogarithmicError(),
                "squared_hinge": tf.keras.losses.SquaredHinge(),
                "hinge": tf.keras.losses.Hinge(), 
                "categorical_hinge": tf.keras.losses.CategoricalHinge(),
                "logcosh": tf.keras.losses.LogCosh(),
                "huber_loss": tf.keras.losses.Huber(),
                "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
                "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy(),
                "kullback_leibler_divergence": tf.keras.losses.KLDivergence(),
                "poisson": tf.keras.losses.Poisson(),
                "cosine_proximity": tf.keras.losses.CosineSimilarity()}

    opts.tf_loss_func = lossMap[opts.lossFunction]

    # See https://keras.io/optimizers/. Also https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/
    optList = { "sgd": tf.keras.optimizers.SGD,
                "rmsprop": tf.keras.optimizers.RMSprop,
                "adagrad": tf.keras.optimizers.Adagrad,
                "adadelta": tf.keras.optimizers.Adagrad,
                "adam": tf.keras.optimizers.Adam,
                "adamax": tf.keras.optimizers.Adamax,
                "nadam": tf.keras.optimizers.Nadam,
                "adamw": tfa.optimizers.AdamW
              }
    if opts.optimizer not in optList.keys():
        msg = "Unsupported loss function %s. Please select on of the following:%s\n\t%s" % (opts.optimizer, ss, "\n\t".join(optList))
        raise Exception(es + msg + ns)
    opts.optimizer_class = optList[opts.optimizer]

    # Get some basic information
    opts.keras      = tf.keras.__version__
    opts.tensorflow = tf.__version__
    opts.hostname   = socket.gethostname()
    opts.python     = "%d.%d.%d" % (sys.version_info[0], sys.version_info[1], sys.version_info[2])
    opts.gitBranch  = subprocess.check_output(["git", "branch", "-a"])
    opts.gitStatus  = subprocess.check_output(["git", "status"])
    opts.gitDiff    = subprocess.check_output(["git", "diff"])

    # Call the main function
    Print("Hostname is %s" % (hs + opts.hostname + ns), True)
    Print("Using Keras %s" % (hs + opts.keras + ns), False)
    Print("Using Tensorflow %s" % (hs + opts.tensorflow + ns), False)

    main(opts)

    Print("Directory %s created" % (ls + opts.saveDir + ns), True)
