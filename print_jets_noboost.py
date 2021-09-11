import argparse
import json

parser = argparse.ArgumentParser(description='Plot jets')
parser.add_argument('model_dir')
parser.add_argument('--model_file')
parser.add_argument('--img_prefix')
parser.add_argument('--utils')
parser.add_argument('--center',action="store_true")
parser.add_argument('--parton',action="store_true")
parser.add_argument('--numplotaxes',default=2,type=int)
parser.add_argument('--setcode',default=None,type=int)
parser.add_argument('--setcodeval',default=1,type=float)
parser.add_argument('--data_path',default='/scratch/jcollins')

args = parser.parse_args()
print(args)
model_dir = args.model_dir
vae_args_file = model_dir + "/vae_args.dat"


with open(vae_args_file,'r') as f:
  vae_arg_dict = json.loads(f.read())

print("\n\n vae_arg_dict:", vae_arg_dict)
latent_dim_lin = vae_arg_dict['latent_dim']
use_vm = False
if 'latent_dim_vm' in vae_arg_dict:
  latent_dim_vm = vae_arg_dict['latent_dim_vm']
  use_vm = True
else:
  latent_dim_vm = 0
latent_dim = latent_dim_lin + latent_dim_vm


if args.model_file:
  if args.model_file == 'last':
    print("Using latest file")
    model_file=None
  model_fn = args.model_file
  model_file = model_dir + '/' + model_fn
  print("Using model file", model_file)
else:
  print("No model file specified, will default to latest file")
  model_file=None




if args.img_prefix:
  file_prefix = args.img_prefix
else:
  file_prefix = model_dir + '/'
print("Saving files with prefix", file_prefix)



import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
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
#from scipy import linalg as LA

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# from colorspacious import cspace_converter
from collections import OrderedDict


from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class

import pandas

#import h5py
#import pickle
from scipy.stats import gaussian_kde

from pyjet import cluster
import re
import glob


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")


if args.utils:
  import importlib
  VAE_model_tools = importlib.import_module(args.utils)
  build_and_compile_annealing_vae = getattr(VAE_model_tools,'build_and_compile_annealing_vae')
  betaVAEModel = getattr(VAE_model_tools,'betaVAEModel')
  reset_metrics = getattr(VAE_model_tools,'reset_metrics')
else:
  import utils.VAE_model_tools_vm
  from utils.VAE_model_tools_vm import build_and_compile_annealing_vae, betaVAEModel, reset_metrics

def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

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

    

def kl_loss(z_mean, z_log_var):
    return -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
    

def get_clustered_pt_eta_phi(pts, locations,R=0.1):
    weights = pts
    outjet = locations
    myjet = np.zeros((weights.shape[-1]),dtype=([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')]))
    myjet['pT'] = weights
    myjet['eta'] = outjet[:,0]
    myjet['phi'] = outjet[:,1]
    sequence = cluster(myjet,R=R,p=0)
    jets = sequence.inclusive_jets()
    phis = np.array([np.mod(np.pi+jet.phi,2*np.pi)-np.pi for jet in jets])
#     phis = [jet.phi for jet in jets]
    etas = np.array([jet.eta for jet in jets])
    pts = np.array([jet.pt for jet in jets])
    
    return pts, etas, phis


def plot_jets(outs_array, numplot = 3, R=0.02,size=50):
    etalim=5
    #bins=np.linspace(-lim, lim, 126)

    for i in range(numplot):   

        fig, ax = plt.subplots(1, 3,figsize=[15,5],sharey=True)



        outjet = valid_y[i,:,1:]
        weights = valid_y[i,:,0]
        pts, etas, phis = get_clustered_pt_eta_phi(weights, outjet,R=R)
        ax[0].scatter(phis, etas, s = pts*size, alpha = 0.7,linewidths=0)
        ax[0].set_title('Jet'+str(i),y=0.9)

        #ax[0].hist2d(feed_pc[i][:,0],feed_pc[i][:,1],range=[[-lim,lim],[-lim,lim]],bins=bins, norm=LogNorm(0.5, 1000))
        for j in range(2):
            outjet = outs_array[j][0][i,:,1:]
            weights = outs_array[j][0][i,:,0]
            pts, etas, phis = get_clustered_pt_eta_phi(weights, outjet,R=R)
            ax[j+1].scatter(phis, etas, s = pts*size, alpha = 0.7,linewidths=0)
            ax[j+1].set_title('Sample'+ str(j),y=0.9)
            
        for j in range(3):
            ax[j].set_ylabel(r'$\eta$',fontsize=18)
            ax[j].set_xlabel(r'$\phi$',fontsize=18)
            ax[j].set_ylim([-0.7,0.7])
            ax[j].set_xlim([-0.7,0.7])

        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.show()
        
def plot_KL_logvar(outs_array,xlim=None,ylim=None,showhist=False, numhists=10,hist_ylim=None,hist_xlim=None):
    
    if use_vm:
      y_pred ,z_mean, z_log_var, losses, _ = outs_array[0]
      KL = losses
    else:
      y_pred ,z_mean, z_log_var, _ = outs_array[0]
      KL=kl_loss(z_mean, z_log_var)

    sort_kl = np.flip(np.argsort(np.mean(KL,axis=0)))

    rms_mean = np.sqrt(np.mean(np.square(z_mean),axis=0))

    plt.scatter(np.mean(KL,axis=0),rms_mean,s=5.)

    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
        
    plt.xlabel('KL divergence')
    plt.ylabel(r'$\sqrt{\left\langle \mu^2 \right\rangle}$')
    #plt.show()
    
    if showhist:
#         for i in range(10):
        plt.hist(np.array(KL)[:,sort_kl[:numhists]],bins=np.linspace(0,20,80),stacked=True)
        #plt.show()
        if hist_ylim:
            plt.ylim(hist_ylim)
        if hist_xlim:
            plt.xlim(hist_xlim)
    
    return sort_kl



# path to file
if args.parton:
  fn =  args.data_path + '/monoW-data-parton.h5'
  numparts = 2
  numtrain = 1500000
  print("Using parton data")
else:
  fn =  args.data_path + '/monoW-data-3.h5'
  numparts = 50
  numtrain = 500000
  print("Using particle data")


df = pandas.read_hdf(fn,stop=100000)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3)+sum(df.memory_usage(deep=True)) / (1024**3))

data = df.values.reshape((-1,numparts,4))
if args.center:
  data = center_jets_ptetaphiE(data)

HT = np.sum(data[:,:,0],axis=-1)
data[:,:,0] = data[:,:,0]/HT[:,None]
data[:,:,-1] = data[:,:,-1]/HT[:,None]


if vae_arg_dict["num_inputs"] == 5:
   sig_input = np.zeros((len(data),numparts,5))
   sig_input[:,:,:2] = data[:,:,:2]
   sig_input[:,:,2] = np.cos(data[:,:,2])
   sig_input[:,:,3] = np.sin(data[:,:,2])
   sig_input[:,:,4] = (np.log(data[:,:,0]+1e-4)/np.log(1e+4) + 1)
else:
   sig_input = np.zeros((len(data),numparts,4))
   sig_input[:,:,:2] = data[:,:,:2]
   sig_input[:,:,2] = np.cos(data[:,:,2])
   sig_input[:,:,3] = np.sin(data[:,:,2])

data_x = sig_input
data_y = data[:,:,:3]


train_x = data_x[:50000]
train_y = data_y[:50000]
valid_x = data_x[50000:]
valid_y = data_y[50000:]


train_output_dir = model_dir #create_dir(osp.join(output_dir, experiment_name))


vae, encoder, decoder = build_and_compile_annealing_vae(**vae_arg_dict)


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

epoch_string=re.compile('_\d*_')
beta_string=re.compile('\d\.[\w\+-]*')
fn_string=re.compile('model_weights_end.*')

def get_epoch(file):
    epoch = int(epoch_string.search(file).group()[1:-1])
    return epoch

def get_beta(file):
    beta = float(beta_string.search(file).group())
    return beta

#def get_fn(file):
#  fn = fn_string.search(file).group()

if model_file is None:
  files = glob.glob(train_output_dir + '/model_weights_end*.hdf5')
  files.sort(key=os.path.getmtime)
  model_file = files[-1]


print("Loading file:", model_file)
beta = get_beta(model_file)
epoch = get_epoch(model_file)
vae.load_weights(model_file)

plt_title = epoch_string.search(model_file).group()[1:-1] + ',    ' + beta_string.search(model_file).group()


outs_array = [vae.predict(valid_x[:1000]) for j in range(3)]

vae.beta.assign(beta)
result = vae.test_step([valid_x[:2000].astype(np.float32),valid_y[:2000].astype(np.float32)])

print("Epoch:", epoch)
print("Beta:", beta)
print("Loss:", "{:.02e}".format(result['loss'].numpy()))
print("Recon loss:", "{:.02e}".format(result['recon_loss'].numpy()))
print("KL loss:", result['KL loss'].numpy())

print()


if args.center:
  plotlim = 0.5
else:
  plotlim = 3.14

plt.figure()
plt.title(plt_title)
sort_kl = plot_KL_logvar(outs_array,[-0.1,None],[-0.1,None])
plt.savefig(file_prefix + plt_title + 'KL_scatter.png')
#plt.show()

plt.figure()
plt.title(plt_title)
plot_jets(outs_array,R=0.02,size=100)
plt.savefig(file_prefix + plt_title + 'recons.png')
#plt.show()

if use_vm:
  _, z_mean, z_log_var, losses, z = outs_array[0]
else:
  _, z_mean, z_log_var, z = outs_array[0]

plt.figure()
plt.title(plt_title)
plt.scatter(z_mean[:,sort_kl[0]],z_mean[:,sort_kl[1]],s=1.)
plt.xlabel(r'$\sqrt{\left\langle \mu_0^2 \right\rangle}$')
plt.ylabel(r'$\sqrt{\left\langle \mu_1^2 \right\rangle}$')
plt.savefig(file_prefix + plt_title + 'scatter_mu0mu1.png')
#plt.show()

if latent_dim > 4:
  plt.figure()
  plt.title(plt_title)
  plt.scatter(z_mean[:,sort_kl[2]],z_mean[:,sort_kl[3]],s=1.)
  plt.xlabel(r'$\sqrt{\left\langle \mu_2^2 \right\rangle}$')
  plt.ylabel(r'$\sqrt{\left\langle \mu_3^2 \right\rangle}$')
  plt.savefig(file_prefix + plt_title + 'scatter_mu2mu3.png')
  #plt.show()

plt.figure()
plt.title(plt_title)
plt.scatter(z_mean[:,sort_kl[0]],z_log_var[:,sort_kl[0]],s=1.)
plt.xlabel(r'$\sqrt{\left\langle \mu_0^2 \right\rangle}$')
plt.ylabel(r'$\left\langle \log \sigma_0 \right\rangle$')
plt.savefig(file_prefix + plt_title + 'scatter_mu0sigma0.png')
#plt.show()

plt.figure()
plt.title(plt_title)
plt.scatter(z_mean[:,sort_kl[1]],z_log_var[:,sort_kl[1]],s=1.)
plt.xlabel(r'$\sqrt{\left\langle \mu_1^2 \right\rangle}$')
plt.ylabel(r'$\left\langle \log \sigma_1 \right\rangle$')
plt.savefig(file_prefix + plt_title + 'scatter_mu1sigma1.png')
#plt.show()

narray = 9
lim = 3.14

for k in range(args.numplotaxes -1):
  codes = np.zeros((narray**2,latent_dim))

  dirs = [0+k,1+k]

  for i in range(narray):
    for j in range(narray):
      codes[narray*i+j,sort_kl[dirs[0]]] = (i-(narray-1)/2)*lim/((narray-1)/2)
      codes[narray*i+j,sort_kl[dirs[1]]] = (j-(narray-1)/2)*lim/((narray-1)/2)
      
  if (args.setcode is not None):
    if (dirs[0] != args.setcode) and (dirs[1] != args.setcode):
      print("Setting code to", args.setcodeval,"for dim", args.setcode)
      codes[:,sort_kl[args.setcode]] = args.setcodeval
      print(codes[0,sort_kl])

  decoded = decoder.predict(codes)

  # fig, ax = plt.subplots(narray, narray,figsize=[15,15],sharex=True,sharey=True)
  fig = plt.figure(figsize=[15,15])
  plt.title(plt_title)
  if args.center:
    circles = [[plt.Circle((i*2/(narray-1)*lim-lim, j*2/(narray-1)*lim-lim), 0.7/(narray-1)*lim,
                       color='black',#[0.8,0.8,0.8],
                       fill=False) for j in range(narray)] for i in range(narray)]

  this = gaussian_kde([z_mean[:,sort_kl[dirs[0]]],z_mean[:,sort_kl[dirs[1]]]],bw_method=0.15)
  xmin=-3.5
  xmax=3.5
  ymin=-3.5
  ymax=3.5
  X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
  positions = np.vstack([X.ravel(), Y.ravel()])
  Z = np.reshape(this(positions).T, X.shape)
  #Z = np.reshape(this(positions).T, X.shape)
  plt.contourf(X,Y,Z,[0.01,0.03,0.05,0.1,0.2,0.3],cmap='Blues',
               vmax=0.4)
  if args.center:
    zoom = 1.
  else:
    zoom = 1/3.14
  for i in range(narray):
    for j in range(narray):
      outjet = decoded[narray*i+j,:,1:]
      weights = decoded[narray*i+j,:,0]
      pts, etas, phis = get_clustered_pt_eta_phi(weights, outjet,R=0.075)
      
      plt.scatter((phis*zoom + 2.*i)/(narray-1)*lim-lim, (etas*zoom + 2.*j)/(narray-1)*lim-lim,
                  s = pts*100, alpha = 1.,linewidths=0,color='tab:red')
      if args.center:
        plt.gcf().gca().add_artist(circles[i][j])
        #         ax[i,j].set_title('['+'{:.1f}'.format(x)+','+'{:.1f}'.format(y)+']',
        
        #         ax[j,i].set_aspect('equal')

    # ax[int((narray-1)/2)-1,int((narray-1)/2)+1].set_facecolor([0.9,0.9,0.9])

  outer_plotlim = -(-0.5-(narray-1)/2)*lim/((narray-1)/2)
  if not args.center:
    plt.vlines(np.linspace(-outer_plotlim,outer_plotlim,narray+1)[1:-1],-outer_plotlim,outer_plotlim,colors='gray')
    plt.hlines(np.linspace(-outer_plotlim,outer_plotlim,narray+1)[1:-1],-outer_plotlim,outer_plotlim,colors='gray')
  plt.xlim([-outer_plotlim,outer_plotlim])
  plt.ylim([-outer_plotlim,outer_plotlim])

  # plt.subplots_adjust(wspace=0, hspace=0)
  # plt.axis('off')
  plt.savefig(file_prefix + plt_title + 'jets_nice_array_' + str(k) + str(k+1) + '.png')
  plt.close()

print('Finished succesfully')
