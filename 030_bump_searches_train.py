

import numpy as np
import pandas as pd

from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras import layers as KL
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, Callback, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.constraints import max_norm
from numpy.random import seed

from sklearn.externals.joblib import dump, load

import json
import pickle
import os
import shutil
import random as rn

from sklearn import preprocessing

from config import *
from vae_utility import *

np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)


######DEFINE FUNCTIONS AND CLASS TO BUILD THE MODEL######

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

def IndividualRecoProb_forVAE_lognorm_5(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0

    mu = par1[:,4:5]
    sigma = par2[:,4:5]
    fraction = par3[:,4:5]
    x_clipped = K.clip(x[:,4:5], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,4:5], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss

def IndividualRecoProb_forVAE_lognorm_6(x, par1, par2, par3, w):
    N = 0
    nll_loss = 0

    mu = par1[:,5:6]
    sigma = par2[:,5:6]
    fraction = par3[:,5:6]
    x_clipped = K.clip(x[:,5:6], clip_x_to0, 1e8)
    single_NLL = K.tf.where(K.less(x[:,5:6], clip_x_to0),
                            -K.log(fraction),
                                -K.log(1-fraction)
                                + K.log(sigma)
                                + K.log(x_clipped)
                                + 0.5*K.square(K.tf.divide(K.log(x_clipped) - mu, sigma)))
    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss



def individualRecoProb_forVAE_discrete_7(x, par1, par2, w):
    N = Nf_lognorm
    nll_loss = 0

    mu = par1[:,6:7]
    sigma = par2[:,6:7]
    norm_xp = K.tf.divide(x[:,6:7] + 0.5 - mu, sigma)
    norm_xm = K.tf.divide(x[:,6:7] - 0.5 - mu, sigma)
    sqrt2 = 1.4142135624
    single_LL = 0.5*(K.tf.erf(norm_xp/sqrt2) - K.tf.erf(norm_xm/sqrt2))

    norm_0 = K.tf.divide(-0.5 - mu, sigma)
    aNorm = 1 + 0.5*(1 + K.tf.erf(norm_0/sqrt2))
    single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) -K.log(aNorm)

    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss

def individualRecoProb_forVAE_discrete_8(x, par1, par2, w):
    N = Nf_lognorm
    nll_loss = 0

    mu = par1[:,7:8]
    sigma = par2[:,7:8]
    norm_xp = K.tf.divide(x[:,7:8] + 0.5 - mu, sigma)
    norm_xm = K.tf.divide(x[:,7:8] - 0.5 - mu, sigma)
    sqrt2 = 1.4142135624
    single_LL = 0.5*(K.tf.erf(norm_xp/sqrt2) - K.tf.erf(norm_xm/sqrt2))

    norm_0 = K.tf.divide(-0.5 - mu, sigma)
    aNorm = 1 + 0.5*(1 + K.tf.erf(norm_0/sqrt2))
    single_NLL = -K.log(K.clip(single_LL, 1e-10, 1e40)) -K.log(aNorm)

    nll_loss += K.sum(w*single_NLL, axis=-1)

    return nll_loss


class CustomKLLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomKLLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mu, sigma, mu_prior, sigma_prior = inputs
        return KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior)

def IdentityLoss(y_train, NETout):
    return K.mean(NETout)

### AUTOENCODER CLASS TO SAVE AUTOENCODER###

class SaveAutoencoder(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
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

def train_vae(name,name_fold, name_weights, dictionary, w, ind_w, intermediate_dim, act_fun, latent_dim, kernel_max_norm,
              lr, epochs, weight_KL_loss, batch_size, sig_inj=[False]):


    if sig_inj[0] == False:
        mod_ind = True

        back = np.load(train_val_test + '/background_train.npy')
        back_val = np.load(train_val_test + '/background_val.npy')
        #remove weights from the train-val data:
        train = back[:,:-1]
        val = back_val[:,:-1]
        
        print("\x1b[31m\" MODEL INDEPENDENT ""\x1b[0m")
        
        try:
            os.makedirs(model_results_bump_single + name_fold + '/' + name_weights)
        except:
            pass


        with open(model_results_bump_single + name_fold + '/' + name_weights +
                  '/' + 'comps_dict.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print("\x1b[31m\" model {} with selected component: {} and weight {} in loss, individual weights {}""\x1b[0m"\
                .format(name, selected_components, weights, individual_weights))


    elif sig_inj[0] == True:
        mod_ind = False

        sig = sig_inj[1]
       
        print("\x1b[31m\" MODEL DEPENDENT on {}""\x1b[0m".format(sig))
        

        back = np.load(train_val_test_mod_dep + 'bck_sig_{}_30_30_40/background_sig_train_{}.npy'.format(sig,sig))
        back_val = np.load(train_val_test_mod_dep + 'bck_sig_{}_30_30_40/background_sig_val_{}.npy'.format(sig,sig))
        #remove weights from the train-val data:
        train = back[:,:-1]
        val = back_val[:,:-1]
        
        try:
            os.makedirs(model_dep_results_bump_single + sig_inj[1] + '/' + name_fold  + '/' + name_weights)
        except:
            pass


        with open(model_dep_results_bump_single + sig_inj[1] + '/' + name_fold +  '/' + name_weights
                  + '/' + 'comps_dict.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print("\x1b[31m\" model {} with selected component: {} and weight {} in loss, individual weights {}""\x1b[0m"\
            .format(name, selected_components, weights, individual_weights))



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
            return IndividualRecoProb_forVAE_lognorm_1(x, par1, par2, par3, w = ind_w[0])

    class CustomIndividualLogNorLayer_2(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualLogNorLayer_2, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return IndividualRecoProb_forVAE_lognorm_2(x, par1, par2, par3, w = ind_w[1])


    class CustomIndividualLogNorLayer_3(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualLogNorLayer_3, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return IndividualRecoProb_forVAE_lognorm_3(x, par1, par2, par3, w = ind_w[2])

    class CustomIndividualLogNorLayer_4(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualLogNorLayer_4, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return IndividualRecoProb_forVAE_lognorm_4(x, par1, par2, par3, w = ind_w[3])

    class CustomIndividualLogNorLayer_5(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualLogNorLayer_5, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return IndividualRecoProb_forVAE_lognorm_5(x, par1, par2, par3, w = ind_w[4])

    class CustomIndividualLogNorLayer_6(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualLogNorLayer_6, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2, par3 = inputs
            return IndividualRecoProb_forVAE_lognorm_6(x, par1, par2, par3, w = ind_w[5])

    class CustomIndividualTruGauLayer_7(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualTruGauLayer_7, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2 = inputs
            return individualRecoProb_forVAE_discrete_7(x, par1, par2, w = ind_w[6])

    class CustomIndividualTruGauLayer_8(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomIndividualTruGauLayer_8, self).__init__(**kwargs)

        def call(self, inputs):
            x, par1, par2 = inputs
            return individualRecoProb_forVAE_discrete_8(x, par1, par2, w = ind_w[7])


    ########### MODEL ###########
    intermediate_dim = intermediate_dim
    act_fun = act_fun
    latent_dim = latent_dim
    kernel_max_norm = kernel_max_norm
    lr = lr
    epochs = epochs
    weight_KL_loss = weight_KL_loss
    batch_size = batch_size

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

    params = KL.concatenate([T_z_mean, T_z_sigma, L_prior_mean, L_prior_sigma, L_par1, L_par2, L_par3], axis=1)

    L_RecoProb_1 = CustomIndividualLogNorLayer_1(name='RecoNLL_met')([x_DNN_input,L_par1,
                                                                  L_par2,L_par3])

    L_RecoProb_2 = CustomIndividualLogNorLayer_2(name='RecoNLL_mt')([x_DNN_input,L_par1,
                                                                  L_par2,L_par3])

    L_RecoProb_3 = CustomIndividualLogNorLayer_3(name='RecoNLL_mbb')([x_DNN_input,L_par1,
                                                                  L_par2,L_par3])

    L_RecoProb_4 = CustomIndividualLogNorLayer_4(name='RecoNLL_mct2')([x_DNN_input,L_par1,
                                                                  L_par2,L_par3])

    L_RecoProb_5 = CustomIndividualLogNorLayer_5(name='RecoNLL_mlb1')([x_DNN_input,L_par1,
                                                                  L_par2,L_par3])

    L_RecoProb_6 = CustomIndividualLogNorLayer_6(name='RecoNLL_lep1Pt')([x_DNN_input,L_par1,
                                                                   L_par2,L_par3])

    L_RecoProb_7 = CustomIndividualTruGauLayer_7(name='RecoNLL_nJet30')([x_DNN_input,L_par1,
                                                                  L_par2])

    L_RecoProb_8 = CustomIndividualTruGauLayer_8(name='RecoNLL_nBJet30_MV2c10')([x_DNN_input,L_par1,
                                                                  L_par2])


    L_RecoProb = CustomRecoProbLayer(name='RecoNLL')([x_DNN_input, L_par1, L_par2, L_par3])
    L_KLLoss = CustomKLLossLayer(name='KL')([T_z_mean, T_z_sigma, L_prior_mean, L_prior_sigma])
    vae = Model(inputs=x_DNN_input, outputs=[L_KLLoss, L_RecoProb,
                                            L_RecoProb_1, L_RecoProb_2,
                                            L_RecoProb_3, L_RecoProb_4, L_RecoProb_5,
                                            L_RecoProb_6, L_RecoProb_7, L_RecoProb_8])

    adam = optimizers.adam(lr)

#         def metric_wapper(layer):

#             N = 4*latent_dim

#             par1 = layer[:, N: N+original_dim]
#             N += original_dim

#             par2 = layer[:, N: N+original_dim]
#             N += original_dim

#             par3 = layer[:, N:N+Nf_lognorm]

#             mu = layer[:, :latent_dim]
#             sigma = layer[:, latent_dim: 2*latent_dim]
#             mu_prior = layer[:, 2*latent_dim: 3*latent_dim]
#             sigma_prior = layer[:, 3*latent_dim: 4*latent_dim]

#             def metric(y_true, y_pred):

#                 KL = weight_KL_loss*KL_loss_forVAE(mu,sigma,mu_prior,sigma_prior)
#                 RecoLoss = RecoProb_forVAE(y_true[:, :original_dim], par1, par2, par3, w=1)

#                 return (K.mean(RecoLoss+KL))

#             return metric

#         metric = metric_wapper(params)

    vae.compile(optimizer=adam,
                loss=[IdentityLoss, IdentityLoss,
                      IdentityLoss, IdentityLoss,IdentityLoss,IdentityLoss,
                     IdentityLoss,IdentityLoss,IdentityLoss,IdentityLoss],
                loss_weights=[weight_KL_loss, 1.,
                              0, 0, 0, 0,
                             0, 0, 0, 0]
#                     , metrics=[metric]
               )

    if mod_ind:

        train_history = vae.fit(x=train, y=[train, train,
                                        train, train, train, train, train, train, train, train],
                                validation_data = (val, [val, val,
                                 val, val, val, val, val, val, val, val]),
                                shuffle=True,
                                epochs=epochs,
                                batch_size=batch_size,

        callbacks = [TerminateOnNaN(),
                    ModelCheckpoint(model_results_bump_single + '{}/{}/vae.h5'.format(name_fold
                                                                                         , name_weights),
                                        monitor='val_loss',
                                        mode='auto', save_best_only=True,verbose=1,
                                        period=1),
                        EarlyStopping(monitor='val_loss', patience=50, verbose=1, min_delta=0.3),
                        ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.8,
                                          patience=5,
                                          mode = 'auto',
                                          epsilon=0.01,
                                          cooldown=0,
                                          min_lr=9e-8,
                                          verbose=1),
                    SaveAutoencoder(model_results_bump_single +'{}/{}/autoencoder.h5'.format(name_fold
                                                                                                ,name_weights),
                                        monitor='val_loss',
                                        mode='auto', save_best_only=True,verbose=1,
                                        period=1)
                                        ])

        hist_df = pd.DataFrame(train_history.history)

        # or save to csv:
        hist_csv_file = model_results_bump_single +'{}/{}/history.csv'.format(name_fold, name_weights)
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

    else:

        train_history = vae.fit(x=train, y=[train, train,
                                        train, train, train, train, train, train, train, train],
                                validation_data = (val, [val, val,
                                 val, val, val, val, val, val, val, val]),
                                shuffle=True,
                                epochs=epochs,
                                batch_size=batch_size,
        callbacks = [TerminateOnNaN(),
                    ModelCheckpoint(model_dep_results_bump_single + sig_inj[1] +  '/{}/{}/vae.h5'.format(
                                                                              name_fold, name_weights),
                                        monitor='val_loss',
                                        mode='auto', save_best_only=True,verbose=1,
                                        period=1),
                        EarlyStopping(monitor='val_loss', patience=50, verbose=1, min_delta=0.3),
                        ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.8,
                                          patience=5,
                                          mode = 'auto',
                                          epsilon=0.01,
                                          cooldown=0,
                                          min_lr=9e-8,
                                          verbose=1),
                    SaveAutoencoder(model_dep_results_bump_single + sig_inj[1]+'/{}/{}/autoencoder.h5'.format(name_fold, name_weights),
                                        monitor='val_loss',
                                        mode='auto', save_best_only=True,verbose=1,
                                        period=1)
                                        ])

        hist_df = pd.DataFrame(train_history.history)

        # or save to csv:
        hist_csv_file = model_dep_results_bump_single + sig_inj[1] + '/{}/{}/history.csv'.format(name_fold, name_weights)
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

            
if __name__ == "__main__":

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)


    cols = ['met', 'mt', 'mbb', 'mct2','mlb1',
            'lep1Pt', 'nJet30','nBJet30_MV2c10'
            ]

    components_dict = {
        'met':5,
        'mt':10,
        'mbb':1,
        'mct2':10,
        'mlb1':1,
        'lep1Pt':1,
        'nJet30':1,
        'nBJet30_MV2c10':1
        }


    selected_components = []

    for k,v in components_dict.items():
        if components_dict[k] != 0:
            selected_components.append(k)

    weights = []

    for k,v in components_dict.items():
        weights.append(v)

    selected_idx = [cols.index(component) for component in selected_components]

#     if you want to train andvalidate on specific column decomment the following
#     lines:

#     train = train[:[selected_idx]]
#     val = val[:[selected_idx]]

#     original_dim = len(selected_components)

#     Nf_PDgauss = 0
#     if 'nJet30' in selected_components:
#         Nf_PDgauss += 1
#     if 'nBJet30_MV2c10' in selected_components:
#         Nf_PDgauss += 1

#     Nf_lognorm = original_dim - Nf_PDgauss


    original_dim = 8
    latent_dim = 3
    Nf_lognorm = 6
    Nf_PDgauss = 2
    #################### WEIGHTS Definistion ###############################

    if 0 in weights:
        individual_weights = [x if x != 0 else 1 for x in weights]
    else:
        individual_weights = weights

    name_fold = 'bump_feat_{}_ld_{}'.format('_'.join([str(x) for x in selected_idx]), latent_dim)
    name_weights = 'bump_weights_{}'.format('_'.join([str(x) for x in weights if x != 0]), latent_dim)
    name = 'bump_feat_{}'.format('_'.join([str(x) for x in weights if x != 0]), latent_dim)

    train_vae(name=name,name_fold=name_fold, name_weights = name_weights, dictionary = components_dict,
              w=weights, ind_w = individual_weights,
              intermediate_dim=50, act_fun='relu',latent_dim=latent_dim,kernel_max_norm=500,
              lr=0.003,epochs=2000,weight_KL_loss=0.6, batch_size=200, sig_inj = [False])


# MODEL CONFIGURATION ON SPLIT 30:30:40
# config/act_fun: relu
# config/batch_size: 200
# config/intermediate_dim: 50
# config/kernel_max_norm: 500
# config/latent_dim: 4
# config/lr: 0.003
# config/w: 2
# config/weight_KL_loss: 0.6
# num_train

# train_vae_46
