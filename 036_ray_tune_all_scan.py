import numpy as np
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras import layers as KL
# import colorama
# from colorama import init, Fore, Style, Style
# import uproot
import numpy
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, Callback, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.constraints import max_norm
from sklearn.externals.joblib import dump, load
from numpy.random import seed
import random
import json
import os
import shutil
import random as rn

from ray.tune.integration.keras import TuneReporterCallback
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler, pbt, PopulationBasedTraining

from sklearn import preprocessing

from config import *
from vae_utility import *

np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)


###LOAD DATA AND DEFINE VARIABLES TO BUILD MODEL #####



def KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior):
    kl_loss = K.tf.multiply(K.square(sigma), K.square(sigma_prior))
    kl_loss += K.square(K.tf.divide(mu_prior - mu, sigma_prior))
    kl_loss += K.log(K.tf.divide(sigma_prior, sigma)) -1
    return 0.5 * K.sum(kl_loss, axis=-1)

def RecoProb_forVAE(x, par1, par2, par3, w):

    N = 0
    nll_loss = 0

    if Nf_lognorm != 0:

        for i in range(Nf_lognorm):

            #Log-Normal distributed variables
            mu = par1[:,i:i+1]
            sigma = par2[:,i:i+1]
            fraction = par3[:,i:i+1]
            x_clipped = K.clip(x[:,i:i+1], clip_x_to0, 1e8)
            single_NLL = K.tf.where(K.less(x[:,i:i+1], clip_x_to0),
                                    -K.log(fraction),
                                        -K.log(1-fraction)
                                        + K.log(sigma)
                                        + K.log(x_clipped)
                                        + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
            nll_loss += K.sum(w[i]*single_NLL, axis=-1)

        N += Nf_lognorm

    if Nf_PDgauss != 0:

        for i in range(N, N+Nf_PDgauss):

            mu = par1[:,i:i+1]
            sigma = par2[:,i:i+1]
            norm_xp = K.tf.divide(x[:,i:i+1] + 0.5 - mu, sigma)
            norm_xm = K.tf.divide(x[:,i:i+1] - 0.5 - mu, sigma)
            sqrt2 = 1.4142135624
            single_LL = 0.5*(K.tf.erf(norm_xp/sqrt2) - K.tf.erf(norm_xm/sqrt2))

            norm_0 = K.tf.divide(-0.5 - mu, sigma)
            aNorm = 1 + 0.5*(1 + K.tf.erf(norm_0/sqrt2))
            single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) -K.log(aNorm)

            nll_loss += K.sum(w[i]*single_NLL, axis=-1)

    return nll_loss


def IndividualRecoProb_forVAE_lognorm_1(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0

    mu = par1[:,:1]
    sigma = par2[:,:1]
    fraction = par3[:,:1]
    x_clipped = K.clip(x[:,:1], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,:1], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss

def IndividualRecoProb_forVAE_lognorm_2(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0

    mu = par1[:,1:2]
    sigma = par2[:,1:2]
    fraction = par3[:,1:2]
    x_clipped = K.clip(x[:,1:2], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,1:2], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss

def IndividualRecoProb_forVAE_lognorm_3(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0

    mu = par1[:,2:3]
    sigma = par2[:,2:3]
    fraction = par3[:,2:3]
    x_clipped = K.clip(x[:,2:3], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,2:3], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss



def IdentityLoss(y_train, NETout):
    return K.mean(NETout)


def IndividualRecoProb_forVAE_lognorm_4(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0

    mu = par1[:,3:4]
    sigma = par2[:,3:4]
    fraction = par3[:,3:4]
    x_clipped = K.clip(x[:,3:4], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,3:4], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss



def IdentityLoss(y_train, NETout):
    return K.mean(NETout)


### AUTOENCODER CLASS TO SAVE AUTOENCODER###

monitor = 'val_loss' #In this case I want to imporove the val_loss (weighted differently for each mode
# but I want also store the val_Metric_loss to fari comparison at the end)

class SaveAutoencoder(ModelCheckpoint):
    def __init__(self, filepath, monitor=monitor, verbose=0,
         save_best_only=False, save_weights_only=False, mode='auto', period=1):

        super(ModelCheckpoint, self).__init__()

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:

            warnings.warn('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.' % (mode),
                      RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = -np.Inf
            else:
                    self.monitor_op = np.less
                    self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        autoencoder = Model(inputs=self.model.input,
                                 outputs=[self.model.get_layer('Output_par1').output,
                                         self.model.get_layer('Output_par2').output,
                                          self.model.get_layer('Output_par3').output])


        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            autoencoder.save_weights(filepath, overwrite=True)
                        else:
                            autoencoder.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    autoencoder.save_weights(filepath, overwrite=True)
                else:
                    autoencoder.save(filepath, overwrite=True)



class TuneReporterCallbackMetrics(TuneReporterCallback, ModelCheckpoint):
    """Tune Callback for Keras."""
    #monito = 'val_loss'
    def __init__(self, reporter=None, freq="batch", logs={}, monitor=monitor,
                 mode = 'auto',batch_size = 1000, num_val_batch = 20, how_many = 5, len_history=3):
        """Initializer.
        Args:
            reporter (StatusReporter|tune.track.log|None): Tune object for
                returning results.
            freq (str): Sets the frequency of reporting intermediate results.
                One of ["batch", "epoch"].
        """
        
        self.reporter = reporter or track.log
        self.iteration = 0
        self.monitor = monitor
        if freq not in ["batch", "epoch"]:
            raise ValueError("{} not supported as a frequency.".format(freq))
        self.freq = freq
        self.mode = mode
        self.num_val_batch = num_val_batch
        self.how_many = how_many
        self.len_history = len_history
        
        self.len_batch = batch_size
        self.len_val = len(val)
        self.len_train = len(train)
        
        self.do_val = int(len(train)/(self.how_many*batch_size))    
        self.len_val_batch = int(len(val)/num_val_batch)
        self.history = []
        
        if mode not in ['auto', 'min', 'max']:
            
            warnings.warn('TuneReporterCallbackMetrics mode %s is unknown, '
                      'fallback to min mode.' % (mode),
                      RuntimeWarning)
            mode = 'min'

        if mode == 'min':
            self.current = np.Inf
        elif mode == 'max':
            self.current = -np.Inf

    def on_batch_end(self, batch, logs={}):
        if not self.freq == "batch":
            return
        self.iteration += 1
        for metric in list(logs):

            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]

        if (self.iteration % self.do_val == 0) | (self.iteration==1):

            i = np.random.randint(self.num_val_batch)

            evaluation = self.model.evaluate(val[i*self.len_val_batch:(i+1)*self.len_val_batch,:], #Input
                                             [val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],#KL_Loss
                                              val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],#RecoNLL_loss
                                              
                                              val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],#ind 1
                                              val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],#ind 2
                                              val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],#ind 3
                                              val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],#ind 4
                                              #val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],#Metric
                                                                                                    ]#
                                                                                                    )[5]
         
         # 2 should be the recon_loss (val recon_loss) >>> Metric_inserted >>> Now the metric is the 5 (with 3 cols, or 6 with 4 cols)
            
         # actually : ['loss', 'KL_loss', 'RecoNLL_loss', 'KL_metric', 'RecoNLL_metric']
         # [4] is the metric with no weighted loss >>> equal comparison among all the models


         #   evaluation = self.model.evaluate(val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],
         #                                    [val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],
         #                                     val[i*self.len_val_batch:(i+1)*self.len_val_batch,:],
         #                                     val[i*self.len_val_batch:(i+1)*self.len_val_batch,:]])[3]
         
         #Before we was interested in the [0]=Total loss, now in [3] that is the metric_loss without weights

            if self.iteration==1:
                self.history.append(evaluation)
                self.history=self.history*self.len_history
                self.eval_mean = np.mean(self.history)
                self.reporter(keras_info=logs, mean_accuracy=self.eval_mean)
            else:
                self.temp = self.history.copy()
                self.history[0] = evaluation
                self.history[1:]=self.temp[0:-1]
                self.eval_mean = np.mean(self.history)
                self.reporter(keras_info=logs, mean_accuracy=self.eval_mean)
        else:
            self.reporter(keras_info=logs, mean_accuracy=self.eval_mean)

    def on_epoch_end(self, batch, logs={}):
#         if not self.freq == "epoch":
#             return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]

        evaluation = logs.get(self.monitor)
        self.reporter(keras_info=logs, mean_accuracy=evaluation)
        print('mean_accuray updating: {}'.format(evaluation))

       
# FUNCTION TO BUILD MODEL AND TO RECALL IN TUNE.RUN DURING THEOPTIMIZATION###


def train_vae(config, reporter):
    # https://github.com/tensorflow/tensorflow/issues/32159
    
    components_dict = {
        'met':config["met"],
        'mt':config["mt"],
        'mct2':config["mct2"],
       # 'lep1Pt':config["lep1Pt"],
        }

    weights = []

    for k,v in components_dict.items():
        weights.append(v)    
        
    w = weights

    class CustomKLLossLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomKLLossLayer, self).__init__(**kwargs)

        def call(self, inputs):
            mu, sigma, mu_prior, sigma_prior = inputs
            return KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior)
        
        
    class CustomRecoProbLayer_not_w(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomRecoProbLayer_not_w, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return RecoProb_forVAE(x, par1, par2, par3, w = [1]*len(components_dict.keys()))


    class CustomRecoProbLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomRecoProbLayer, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return RecoProb_forVAE(x, par1, par2, par3, w = w)

    #################################################################################################à
    class CustomIndividualLogNorLayer_1(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualLogNorLayer_1, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return IndividualRecoProb_forVAE_lognorm_1(x, par1, par2, par3, w = w[0])

    class CustomIndividualLogNorLayer_2(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualLogNorLayer_2, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return IndividualRecoProb_forVAE_lognorm_2(x, par1, par2, par3, w = w[1])


    class CustomIndividualLogNorLayer_3(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualLogNorLayer_3, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return IndividualRecoProb_forVAE_lognorm_3(x, par1, par2, par3, w = w[2])


#     class CustomIndividualLogNorLayer_4(Layer):
#         def __init__(self, **kwargs):
#             self.is_placeholder = True
#             super(CustomIndividualLogNorLayer_4, self).__init__(**kwargs)

#         def call(self, inputs):
#             x, par1, par2, par3 = inputs
#             return IndividualRecoProb_forVAE_lognorm_4(x, par1, par2, par3, w = w[3])



    ########### MODEL ###########
    intermediate_dim = config["intermediate_dim"]
    act_fun = config["act_fun"]
    latent_dim = config["latent_dim"]
    kernel_max_norm = config["kernel_max_norm"]
    lr = config["lr"]
    epochs = config["epochs"]
    weight_KL_loss = config["weight_KL_loss"]
    batch_size = config["batch_size"]
    #opt = config['optimizer']
    patience = 50
 

   # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)

    x_DNN_input = Input(shape=(original_dim,), name='Input')
    hidden_1 = Dense(intermediate_dim, activation=act_fun, name='Encoder_h1')
    aux = hidden_1(x_DNN_input)

    hidden_2 = Dense(intermediate_dim, activation=act_fun, name='Encoder_h2')

    aux = hidden_2(aux)

    L_z_mean = Dense(latent_dim, name='Latent_mean')
    T_z_mean = L_z_mean(aux)
    L_z_sigma_preActivation = Dense(latent_dim, name='Latent_sigma_h')

    aux = L_z_sigma_preActivation(aux)
    L_z_sigma = Lambda(InverseSquareRootLinearUnit, name='Latent_sigma')
    T_z_sigma = L_z_sigma(aux)

    L_z_latent = Lambda(sampling, name='Latent_sampling')([T_z_mean, T_z_sigma])
    decoder_h1 = Dense(intermediate_dim,
                       activation=act_fun,
                       kernel_constraint=max_norm(kernel_max_norm),
                       name='Decoder_h1')(L_z_latent)

    decoder_h2 = Dense(intermediate_dim, activation=act_fun, name='Decoder_h2')(decoder_h1)

    L_par1 = Dense(original_dim, name='Output_par1')(decoder_h2)

    L_par2_preActivation = Dense(original_dim , name='par2_h')(decoder_h2)
    L_par2 = Lambda(InverseSquareRootLinearUnit, name='Output_par2')(L_par2_preActivation)

    L_par3_preActivation = Dense(Nf_lognorm, name='par3_h')(decoder_h2)
    L_par3 = Lambda(ClippedTanh, name='Output_par3')(L_par3_preActivation)

    fixed_input = Lambda(SmashTo0)(x_DNN_input)
    h1_prior = Dense(1,
                     kernel_initializer='zeros',
                     bias_initializer='ones',
                     trainable=False,
                     name='h1_prior'
                    )(fixed_input)

    L_prior_mean = Dense(latent_dim,
                         kernel_initializer='zeros',
                         bias_initializer='zeros',
                         trainable=True,
                         name='L_prior_mean'
                        )(h1_prior)

    L_prior_sigma_preActivation = Dense(latent_dim,
                                        kernel_initializer='zeros',
                                        bias_initializer='ones',
                                        trainable=True,
                                        name='L_prior_sigma_preAct'
                                       )(h1_prior)
    L_prior_sigma = Lambda(InverseSquareRootLinearUnit, name='L_prior_sigma')(L_prior_sigma_preActivation)

    L_RecoProb_1 = CustomIndividualLogNorLayer_1(name='RecoNLL_{}'.format(cols[0]))([x_DNN_input,L_par1,
                                                                  L_par2,L_par3])

    L_RecoProb_2 = CustomIndividualLogNorLayer_2(name='RecoNLL_{}'.format(cols[1]))([x_DNN_input,L_par1,
                                                                  L_par2,L_par3])

    L_RecoProb_3 = CustomIndividualLogNorLayer_3(name='RecoNLL_{}'.format(cols[2]))([x_DNN_input,L_par1,
                                                                  L_par2,L_par3])

#    L_RecoProb_4 = CustomIndividualLogNorLayer_4(name='RecoNLL_{}'.format(cols[3]))([x_DNN_input,L_par1,
#                                                                  L_par2,L_par3])


    L_RecoProb = CustomRecoProbLayer(name='RecoNLL')([x_DNN_input, L_par1, L_par2, L_par3])
    L_KLLoss = CustomKLLossLayer(name='KL')([T_z_mean, T_z_sigma, L_prior_mean, L_prior_sigma])
    Metric = CustomRecoProbLayer_not_w(name='Metric')([x_DNN_input, L_par1, L_par2, L_par3])

    vae = Model(inputs=x_DNN_input, outputs=[L_KLLoss, L_RecoProb
                                                 , L_RecoProb_1
                                                 , L_RecoProb_2
                                                 , L_RecoProb_3
                                                # , L_RecoProb_4 
                                                 , Metric
                                                ])

    adam = optimizers.adam(lr)

#     def metric_wapper(layer): 

#         N = 4*latent_dim
    
#         par1 = layer[:, N: N+original_dim]
#         N += original_dim
    
#         par2 = layer[:, N: N+original_dim]
#         N += original_dim
    
#         par3 = layer[:, N:N+Nf_lognorm]
    
#         mu = layer[:, :latent_dim]
#         sigma = layer[:, latent_dim: 2*latent_dim]
#         mu_prior = layer[:, 2*latent_dim: 3*latent_dim]
#         sigma_prior = layer[:, 3*latent_dim: 4*latent_dim]

#         def metric(y_true, y_pred):   
        
#             KL = weight_KL_loss*KL_loss_forVAE(mu,sigma,mu_prior,sigma_prior)
#             RecoLoss = RecoProb_forVAE(y_true[:, :original_dim], par1, par2, par3, w=w)
            
#             return (K.mean(RecoLoss)) # I Remove KL in the sum because i want to compare only on the ReconLoss
        
#         return metric

#     metric = metric_wapper(params)

    vae.compile(optimizer=adam,
                loss=[IdentityLoss, IdentityLoss
                     , IdentityLoss
                     , IdentityLoss
                     , IdentityLoss
                     , IdentityLoss
                    # , IdentityLoss #Metric
                     ],
                loss_weights=[weight_KL_loss, 1.
                             , 0
                             , 0
                             , 0
                             , 0
                            # , 0 #Metric
                             ]
#                 , metrics=[metric]
               )

    fit_report = vae.fit(x=train, y=[train, train
                                    , train
                                    , train
                                    , train
                                    , train
                                   # , train #Metric
                                    ],
        validation_data = (val, [val, val
                                , val
                                , val
                                , val
                                , val
                               # , val #Metric
                                ]),
        shuffle=True,
        epochs=epochs,

        batch_size=batch_size,
        callbacks = [TuneReporterCallbackMetrics(reporter, batch_size = batch_size, num_val_batch = 30, how_many=5),
                    ModelCheckpoint('ray_tuned_{}.h5'.format('vae'),
                                        monitor='val_loss',
                                        mode='auto', save_best_only=True,verbose=1,
                                        period=1),
                        EarlyStopping(monitor='val_loss', patience=patience, verbose=1, min_delta=0.3),
                        ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.8,
                                          patience=5,
                                          mode = 'auto',
                                          epsilon=0.01,
                                          cooldown=0,
                                          min_lr=9e-8,
                                          verbose=1),
                    TerminateOnNaN(),
                    SaveAutoencoder('ray_tuned_{}.h5'.format('autoencoder'),
                                        monitor='val_loss',
                                        mode='auto', save_best_only=True,verbose=1,
                                        period=1)
                                        ])

def main(name_exp = 'exp', overwrite_exp=True):

    epochs = 1000
    batch_size_max = 2000
    batch_size_min = 100

    np.random.seed(42)
    rn.seed(12345)


    if overwrite_exp:
       folders = os.listdir(str(ray_model_results))
       try:
           os.remove(str(ray_model_results + name_exp + '/tune_result.csv'))
       except:
           pass
       
       if name_exp in folders:
          try:
              shutil.rmtree(str(ray_model_results +  name_exp)) 
              print('these are the folders {}'.format(folders))
          except:
              pass

    ray.init()
    sched = ASHAScheduler(
         time_attr="training_iteration",
         metric="mean_accuracy",
         mode="min",
         #max_t=int(len(train)/ batch_size_min * epochs),
         #grace_period=int(len(train)/ batch_size_min * epochs)
         max_t=10**9,
         grace_period=10**9

         )

    analysis = tune.run(
        train_vae,
        local_dir=str(ray_model_results),
        name=name_exp,
         scheduler=sched,
         raise_on_failed_trial=False,
         max_failures=10,
#         stop={
#              "mean_accuracy": 0.99,
#             "training_iteration": 5 if args.smoke_test else 300
#         },
        num_samples=4,
        resources_per_trial={
            "gpu": 0,
            "cpu":8
        },

            config={
            "intermediate_dim": tune.sample_from(lambda spec: random.choice([20,30,50,80,100,120,150])),
            "act_fun": tune.sample_from(lambda spec: random.choice(["elu","relu"])),
            "latent_dim": tune.sample_from(lambda spec: random.choice([2,3,3])),
            "kernel_max_norm": tune.sample_from(lambda spec: random.choice([100,200,500])),
            "lr": tune.sample_from(lambda spec: random.choice([0.003,0.006,0.001,0.009,0.0009])),
            #"optimizer": tune.sample_from(lambda spec: random.choice(['adadelta'])),
            "epochs": epochs,
            "weight_KL_loss": tune.sample_from(lambda spec: random.choice([0.4,0.6,0.8])),
            "batch_size": tune.sample_from(lambda spec: random.choice([200])),
            'met': tune.sample_from(lambda spec: random.choice([1,2,5,10,20])),
            'mt': tune.sample_from(lambda spec: random.choice([1,2,5,10,20])),
            'mct2': tune.sample_from(lambda spec: random.choice([1,2,5,10,20])),
           # 'lep1Pt':tune.sample_from(lambda spec: random.choice([1,2,5,10,20])),      
           })

    df = analysis.dataframe()
    df.to_csv(str(ray_model_results) + name_exp + '/tune_result.csv', index=False)

    ray.shutdown()

if __name__ == "__main__":
            
    np.random.seed(42)
    rn.seed(12345)
   
    datasets = ['wh']
    dataset = datasets[0]

    cols_or = ['met', 'mt', 'mbb', 'mct2','mlb1',
             'lep1Pt', 'nJet30','nBJet30_MV2c10'
             ]

    cols_sel = ['met','mt','mct2']

    name_exp = '{}_{}_feats'.format(dataset, len(cols_sel))

    # TO DO:
    # Add lep1Pt feature
   
    if dataset == 'wh':

        back = np.load(train_val_test + '/background_train.npy',allow_pickle = True)
        back_val = np.load(train_val_test + '/background_val.npy',allow_pickle = True)
        
        print('loading data from {}'.format(train_val_test))

        train_df = pd.DataFrame(back[:,:-1], columns= ['met', 'mt', 'mbb', 'mct2','mlb1','lep1Pt', 'nJet30','nBJet30_MV2c10'])
        
#         train_df = train_df[((train_df['mbb']>=100)&(train_df['mbb']<350))]
#         train_df = train_df[((train_df['mct2']>=100)&(train_df['mct2']<1000))]
#         train_df = train_df[((train_df['mt']>=0)&(train_df['mt']<1000))]
#         train_df = train_df[((train_df['met']>=0)&(train_df['mt']<1000))]
#         train_df = train_df[((train_df['mlb1']>=0)&(train_df['mlb1']<1000))]
#         train_df = train_df[((train_df['lep1Pt']>=0)&(train_df['lep1Pt']<1000))]        
        
        train = train_df[cols_sel].values

        val_df = pd.DataFrame(back_val[:,:-1], columns=['met', 'mt', 'mbb', 'mct2','mlb1','lep1Pt', 'nJet30','nBJet30_MV2c10'])
        
#         val_df = val_df[((val_df['mbb']>=100)&(val_df['mbb']<350))]
#         val_df = val_df[((val_df['mct2']>=100)&(val_df['mct2']<1000))]
#         val_df = val_df[((val_df['mt']>=0)&(val_df['mt']<1000))]
#         val_df = val_df[((val_df['met']>=0)&(val_df['met']<1000))]
#         val_df = val_df[((val_df['mlb1']>=0)&(val_df['mlb1']<1000))]
#         val_df = val_df[((val_df['lep1Pt']>=0)&(val_df['lep1Pt']<1000))]        

        val = val_df[cols_sel].values
    
    Nf_lognorm=0
    Nf_PDgauss=0
    for var in cols_sel:
        if var in cols_or[:-2]:
            Nf_lognorm += 1
        else:
            Nf_PDgauss += 1

    print('lognorm {} and pdgauss {}'.format(Nf_lognorm,Nf_PDgauss))

    original_dim = len(cols_sel)

    main(name_exp = name_exp , overwrite_exp=True)
