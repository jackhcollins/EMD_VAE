import os
import os.path as osp
import sys
import json
import argparse
import glob
import re
import gc
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model_dir')
parser.add_argument('--model_file','--model_fn')
parser.add_argument('--parton',action='store_true')
parser.add_argument('--data_path',default='/scratch/jcollins')
parser.add_argument('--center',action='store_true')
parser.add_argument('--alpha',default=1.,type=float)
parser.add_argument('--time_order',action="store_true")
parser.add_argument('--maxepochs',default=50,type=int)

args = parser.parse_args()
print(args)
model_dir = args.model_dir
vae_args_file = model_dir + "/vae_args.dat"

init_epoch=0
start_i = 0
end_dropout = 120
if args.model_file == 'last':
  files = glob.glob(model_dir + '/model_weights_end*.hdf5')

  import re
  start_i = len(files)
  def get_epoch(file):
    epoch = int(epoch_string.search(file).group()[1:-1])
    return epoch

  def get_beta(file):
    beta = float(beta_string.search(file).group())
    return beta

  epoch_string=re.compile('_\d*_')
  beta_string=re.compile('\d\.[\w\+-]*')

  if args.time_order:
    files.sort(key=os.path.getmtime)
    epochs = np.array([get_epoch(model_file) for model_file in files])
  else:
    epochs = np.array([get_epoch(model_file) for model_file in files])
    sorted_args = np.argsort(epochs)
    files = [files[index] for index in sorted_args]
    epochs = epochs[sorted_args]

  model_file = files[-1]
  with open(vae_args_file,'r') as f:
    vae_arg_dict = json.loads(f.read())

  print("\n\n vae_arg_dict:", vae_arg_dict)

  init_epoch = get_epoch(model_file)

  print("Starting from epoch", init_epoch)#, ", and beta", betas[start_i])

elif args.model_file is not None:
  model_fn = args.model_file
  model_file = model_dir + '/' + model_fn
  print("Using model file", model_file)
  with open(vae_args_file,'r') as f:
    vae_arg_dict = json.loads(f.read())

  print("\n\n vae_arg_dict:", vae_arg_dict)
else:
  print("No model file specified, will train from beginning")
  model_file=None


import tensorflow as tf
# tf.config.experimental.set_visible_devices([], 'GPU')
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

from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
import utils.VAE_model_tools_cat
from utils.VAE_model_tools_cat import build_and_compile_annealing_vae, betaVAEModel, reset_metrics, loss_tracker, myTerminateOnNaN

import pandas

import h5py
import pickle

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")



def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

model_dir = create_dir(model_dir)

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


def Epxpypz_to_ptetaphiE(jets):

  E = jets[:,:,0]
  px = jets[:,:,1]
  py = jets[:,:,2]
  pz = jets[:,:,3]

  pt = np.sqrt(np.square(px) + np.square(py))
  phi = np.arctan2(py,px)
  eta = np.arcsinh(pz/pt)

  newjets = np.zeros(jets.shape)
  newjets[:,:,0] = pt
  newjets[:,:,1] = eta
  newjets[:,:,2] = phi
  newjets[:,:,3] = E

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
    msqr[msqr < 0.] = 0
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
fn =  args.data_path + '/train_processed.h5'
numtrain = 1000000
numparts=50

print("Loading ", fn)
df = pandas.read_hdf(fn,stop=1100000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))

data = df.values[:,1:].reshape((-1,numparts,4))
data[data == 1e5] = 0
#E = data[:,:,0:1]*np.cosh(data[:,:,1:2])
#data = np.concatenate((data,E),axis=-1) 

if args.center:
  data = center_jets_ptetaphiE(data)

# Normalize pTs so that HT = 1
HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
data[:,:,-1] = data[:,:,-1]/HT[:,None]

# Inputs x to NN will be: pT, eta, cos(phi), sin(phi), log E
# Separated phi into cos and sin for continuity around full detector, so make things easier for NN.
# Also adding the log E is mainly because it seems like it should make things easier for NN, since there is an exponential spread in particle energies.
# Feel free to change these choices as desired. E.g. px, py might be equally as good as pt, sin, cos.
sig_input = np.zeros((len(data),numparts,4))
sig_input[:,:,:2] = data[:,:,:2]
sig_input[:,:,2] = np.cos(data[:,:,2])
sig_input[:,:,3] = np.sin(data[:,:,2])
#sig_input[:,:,4] = np.log(data[:,:,3]+1e-8)


data_x = sig_input
# Event 'labels' y are [pT, eta, phi], which is used to calculate EMD to output which is also pT, eta, phi.
data_y = data[:,:,:3]


train_x = data_x[:numtrain]
train_y = data_y[:numtrain]
valid_x = data_x[numtrain:numtrain+100000]
valid_y = data_y[numtrain:numtrain+100000]


#output_dir = '/scratch/jcollins'

#experiment_name = 'W-parton-centered-vm-lin2'
train_output_dir = create_dir(model_dir)
last_save = None

if model_file is None:


  if osp.exists(vae_args_file):
    print("Loading", vae_args_file)
    with open(vae_args_file,'r') as f:
      vae_arg_dict = json.loads(f.read())
  else:
    vae_arg_dict = {"encoder_conv_layers": [1024,1024,1024,1024],
                    "dense_size": 1024,
                    "decoder_sizes": [1024,1024,1024,1024,1024],
                    "numIter": 10,
                    "reg_init": 1.,
                    "reg_final": 0.01,
                    "stopThr": 1e-3,
                    "num_inputs": 4,           # Size of x (e.g. pT, eta, sin, cos, log E)
                    "num_particles_in": numparts,
                    "latent_dim": 256,
                    "verbose": 1,
                    "dropout": 0.,
                    "cat_dim": 4,#}
                    "cat_priors": list(np.logspace(np.log10(0.5)-3,np.log10(0.5),4))}

    print("Saving vae_arg_dict to",vae_args_file)
    print("\n",vae_arg_dict)

    with open(vae_args_file,'w') as file:
      file.write(json.dumps(vae_arg_dict))


  vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)

else:
#  if start_i < end_dropout:
#    vae_arg_dict["dropout"] = 0.1
  vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)
  vae.fit(x=train_x[:1], y=train_y[:1], batch_size=1, epochs=1,verbose=2)
  vae.load_weights(model_file)
  last_save = model_file

batch_size=100
reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=5, verbose=1, mode='auto', min_delta=1e-4, cooldown=0, min_lr=1e-8)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0., patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)
reset_metrics_inst = reset_metrics()

callbacks=[tf.keras.callbacks.CSVLogger(train_output_dir + '/log.csv', separator=",", append=True),
           reduceLR,
           earlystop,
           myTerminateOnNaN(),
           reset_metrics_inst]

beta_set = np.logspace(-5,1,25)[:-7]
betas = beta_set

for i in range(0,10,2):
  betas = np.append(betas, beta_set[-1-7-i:-1-i])

last_run_i = len(betas)
betas = np.append(betas, beta_set)


print(betas)

steps_per_epoch = 1000
save_period = 10
nan_counter = 0

max_epoch_per_step = args.maxepochs
#switch_max_epochs = len(beta_set_init)

vae.alpha.assign(args.alpha)

i = start_i

while i < len(betas):
    beta = betas[i]
#    if i == switch_max_epochs:
#      max_epoch_per_step = 50
#    print("\n Changing beta to", beta)

    vae.beta.assign(beta)

    if i < last_run_i:
      K.set_value(vae.optimizer.lr,3e-5)
    else:
      K.set_value(vae.optimizer.lr,1e-5)

#    K.set_value(vae.optimizer.beta_1,0.99)

    my_history = vae.fit(x=train_x, y=train_y, batch_size=batch_size,
                epochs=init_epoch + max_epoch_per_step,verbose=2,
                         validation_data = (valid_x[:100000],valid_y[:100000]),
                callbacks = callbacks,
                initial_epoch=init_epoch,
                steps_per_epoch = steps_per_epoch
              )

    if np.isnan(loss_tracker.result().numpy()):
      if nan_counter > 10:
        print(nan_counter, "NaNs. Too many. Quitting.")
        quit()
      if last_save:
        print("Went Nan, reloading", last_save)
        nan_counter = nan_counter + 1
        vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)
        vae.fit(x=train_x[:100], y=train_y[:100], batch_size=100, epochs=1,verbose=2)
        vae.load_weights(last_save)
      else:
        print("Went nan but no last save, quitting...")
        quit()
    else:
      init_epoch = my_history.epoch[-1]+1
      i = i+1

    last_save = train_output_dir + '/model_weights_end_' + str(init_epoch) + '_' + "{:.1e}".format(beta) + '.hdf5'
    vae.save_weights(last_save)

    gc.collect()
