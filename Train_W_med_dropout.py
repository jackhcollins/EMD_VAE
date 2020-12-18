import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import tensorflow.keras as keras
import tensorflow.keras.backend as K

import os
import os.path as osp
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
import utils.VAE_model_tools
from utils.VAE_model_tools import build_and_compile_annealing_vae, betaVAEModel, reset_metrics, loss_tracker

import pandas
import matplotlib.pyplot as plt

import h5py
import pickle

def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

output_dir = './data/'

def ptetaphiE_to_Epxpypz(jets):
    pt = jets[:,:,0]
    eta = jets[:,:,1]
    phi = jets[:,:,2]
    E = jets[:,:,3]
    
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = E
    newjets[:,:,1] = px
    newjets[:,:,2] = py
    newjets[:,:,3] = pz
    
    return newjets

def ptetaphiE_to_ptyphim(jets):
    pt = jets[:,:,0]
    eta = jets[:,:,1]
    phi = jets[:,:,2]
    E = jets[:,:,3]
    
    pz = pt * np.sinh(eta)
    y = 0.5*np.nan_to_num(np.log((E+pz)/(E-pz)))
    
    msqr = np.square(E)-np.square(pt)-np.square(pz)
    msqr[np.abs(msqr) < 1e-6] = 0
    m = np.sqrt(msqr)
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = pt
    newjets[:,:,1] = y
    newjets[:,:,2] = phi
    newjets[:,:,3] = m
    
    return newjets
    
def ptyphim_to_ptetaphiE(jets):
    
    pt = jets[:,:,0]
    y = jets[:,:,1]
    phi = jets[:,:,2]
    m = jets[:,:,3]
    
    eta = np.nan_to_num(np.arcsinh(np.sinh(y)*np.sqrt(1+np.square(m/pt))))
    pz = pt * np.sinh(eta)
    E = np.sqrt(np.square(pz)+np.square(pt)+np.square(m))
    
    newjets = np.zeros(jets.shape)
    newjets[:,:,0] = pt
    newjets[:,:,1] = eta
    newjets[:,:,2] = phi
    newjets[:,:,3] = E
    
    return newjets
    
def center_jets_ptetaphiE(jets):
    cartesian_jets = ptetaphiE_to_Epxpypz(jets)
    sumjet_cartesian = np.sum(cartesian_jets,axis=1)
    
    sumjet_phi = np.arctan2(sumjet_cartesian[:,2],sumjet_cartesian[:,1])
    sumjet_y = 0.5*np.log((sumjet_cartesian[:,0] + sumjet_cartesian[:,-1])/(sumjet_cartesian[:,0] - sumjet_cartesian[:,-1]))
    
    ptyphim_jets = ptetaphiE_to_ptyphim(jets)
    #print(ptyphim_jets[:3,:,:])
    
    transformed_jets = np.copy(ptyphim_jets)
    transformed_jets[:,:,1] = ptyphim_jets[:,:,1] - sumjet_y[:,None]
    transformed_jets[:,:,2] = ptyphim_jets[:,:,2] - sumjet_phi[:,None]
    transformed_jets[:,:,2] = transformed_jets[:,:,2] + np.pi
    transformed_jets[:,:,2] = np.mod(transformed_jets[:,:,2],2*np.pi)
    transformed_jets[:,:,2] = transformed_jets[:,:,2] - np.pi

    transformed_jets[transformed_jets[:,:,0] == 0] = 0
    
    newjets = ptyphim_to_ptetaphiE(transformed_jets)
    return newjets
    
# path to file
fn =  'E:\projects\EMD_VAE\in_data\monoW-data-3.h5'

df = pandas.read_hdf(fn,stop=1000000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))

data = df.values.reshape((-1,50,4))

HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
data[:,:,-1] = data[:,:,-1]/HT[:,None]

data = center_jets_ptetaphiE(data)

sig_input = np.zeros((len(data),50,5))
sig_input[:,:,:2] = data[:,:,:2]
sig_input[:,:,2] = np.cos(data[:,:,2])
sig_input[:,:,3] = np.sin(data[:,:,2])
sig_input[:,:,4] = np.log(data[:,:,3]+1e-8)

data_x = sig_input
data_y = data[:,:,:3]


train_x = data_x[:500000]
train_y = data_y[:500000]
valid_x = data_x[500000:600000]
valid_y = data_y[500000:600000]

experiment_name = 'W-med-dropout'
train_output_dir = create_dir(osp.join(output_dir, experiment_name))
vae, encoder, decoder = build_and_compile_annealing_vae(optimizer=keras.optimizers.Adam(lr=0.001,clipnorm=0.1),
                                    encoder_conv_layers = [2048,2048,1024,1024],
                                    dense_size = [1024,1024,1024,1024],
                                    decoder = [4096,2048,1024,1024,1024],
                                    numItermaxinner = 10,
                                    numIter=10,
                                    reg_init = 1.,
                                    reg_final = 0.01,
                                    latent_dim=128,
                                    stopThr=1e-3,
                                    num_inputs=5,
                                    num_particles_in=50,
                                    dropout = 0.1)

batch_size=100
save_period=2

vae.beta.assign(0.001)

K.set_value(vae.optimizer.lr,1e-4)
epochs = 1


history = vae.fit(x=train_x[:10], y=train_y[:10], batch_size=batch_size,
                epochs=epochs,verbose=1,#initial_epoch=int(vae.optimizer.iterations/numbatches),
                validation_data = (valid_x[:10],valid_y[:10]),
                callbacks = None
              )


# init_epoch = 1637
# # init_alpha = 1.0e-8
# init_beta = 9.1e-4
# last_save = train_output_dir + '/model_weights_end_' + str(init_epoch) + "_" + "{:.1e}".format(init_beta) + '.hdf5'
# vae.load_weights(last_save)


# init_epoch = 237
steps_per_epoch=1000
save_period = 10
beta = 0.1
init_epoch=0
betas = np.concatenate((np.logspace(-3,np.log10(4e-2),10),
                            np.logspace(np.log10(4e-2),-3,10)[1:],
                            np.logspace(-3,np.log10(4e-2),20)[1:],
                            np.logspace(np.log10(4e-2),-4,20)[1:],
                            np.logspace(-4,np.log10(4e-2),30)[1:]))

i = 0
for i in range(0,len(betas)):

    beta = betas[i]

    print('\n Setting beta =', str(beta))
    
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5, verbose=1, mode='auto', min_delta=1e-4, cooldown=0, min_lr=1e-8)
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0., patience=10, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )

    # modelcheckpoint = keras.callbacks.ModelCheckpoint(train_output_dir + '/model_weights_{epoch:02d}_' + "{:.1e}".format(beta) + '.hdf5', save_freq = save_period*steps_per_epoch, save_weights_only=True)
    reset_metrics_inst = reset_metrics()

    callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
                reduceLR,earlystop,
                #modelcheckpoint,
                reset_metrics_inst]

    vae.beta.assign(beta)
    K.set_value(vae.optimizer.lr,3e-5)
    

    my_history = vae.fit(x=train_x, y=train_y, batch_size=batch_size,
                epochs=init_epoch+300,verbose=1,#initial_epoch=int(vae.optimizer.iterations/numbatches),
                validation_data = (valid_x[:200*batch_size],valid_y[:200*batch_size]),
                callbacks = callbacks,initial_epoch=init_epoch,steps_per_epoch=steps_per_epoch
            )

    if np.isnan(loss_tracker.result().numpy()):
        vae, encoder, decoder = build_and_compile_annealing_vae(optimizer=keras.optimizers.Adam(lr=0.001,clipnorm=0.1),
                                    encoder_conv_layers = [2048,2048,2048,2048],
                                    dense_size = [2048,1024,1024,1024],
                                    decoder = [4096,2048,2048,1024,1024],
                                    numItermaxinner = 10,
                                    numIter=10,
                                    reg_init = 1.,
                                    reg_final = 0.01,
                                    latent_dim=128,
                                    stopThr=1e-3,
                                    num_inputs=5,
                                    num_particles_in=50)

        history = vae.fit(x=train_x[:10], y=train_y[:10], batch_size=batch_size,
                epochs=1,verbose=1,#initial_epoch=int(vae.optimizer.iterations/numbatches),
                validation_data = (valid_x[:10],valid_y[:10]),
                callbacks = callbacks
              )
        
        vae.load_weights(last_save)
    else:
        init_epoch = my_history.epoch[-1]+1
        i = i+1


    last_save = train_output_dir + '/model_weights_end_' + str(init_epoch) + "_" + "{:.1e}".format(beta) + '.hdf5'
    vae.save_weights(train_output_dir + '/model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5')
