import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import numpy as np


tfkl = tfk.layers
tfd = tfp.distributions
tfpl = tfp.layers

def make_encoder_oneparticle(data_inputs,
                             dense_sizes = [64,64,64],
                             latent_dims_line = 1,
                             latent_dims_circle = 1,
                             real_dim=2):
    
    layer = data_inputs
    
    for layer_size in dense_sizes:
        layer = tfkl.Dense(layer_size,
                           activation='relu',
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')(layer)
    
    gauss_z_mean = tfkl.Dense(latent_dims_line, name='gauss_z_mean',activation=None)(layer)
    gauss_z_log_var = tfkl.Dense(latent_dims_line, name='gauss_z_log_var',activation=None)(layer)
    
    vm_z_mean_x = tfkl.Dense(latent_dims_circle, name='vm_z_mean_x',activation=None)(layer)
    vm_z_mean_x = tfkl.BatchNormalization(renorm=True,
                                          momentum=0.9999,
                                          renorm_momentum=0.9999,
                                          renorm_clipping={'rmax':3.,'rmin':1./3,'dmax':5.}
                                         )(vm_z_mean_x)
    vm_z_mean_y = tfkl.Dense(latent_dims_circle, name='vm_z_mean_y',activation=None)(layer)
    vm_z_mean_y = tfkl.BatchNormalization(renorm=True,
                                          momentum=0.9999,
                                          renorm_momentum=0.9999,
                                          renorm_clipping={'rmax':3.,'rmin':1./3,'dmax':5.}
                                         )(vm_z_mean_y)
    vm_z_mean = tf.atan2(vm_z_mean_x,vm_z_mean_y)
#     vm_z_mean = tfkl.Dense(latent_dims_circle, name='vm_z_mean_y',activation=None)(layer)
    
    #exp threshold
    up_thresh = 1.0
    low_thresh = -30
    half_thresh_diff = (up_thresh - low_thresh)/2
    cprime = np.arctanh((half_thresh_diff-up_thresh)/half_thresh_diff)
    vm_z_log_var = tfkl.Dense(latent_dims_circle, name='vm_z_log_var',activation=None)(layer)
    vm_z_log_var = tfkl.Lambda(lambda x:  x/half_thresh_diff+cprime)(vm_z_log_var)
    vm_z_log_var = tfkl.Activation('tanh')(vm_z_log_var)
    vm_z_log_var = tfkl.Lambda(lambda x:  x*half_thresh_diff + up_thresh - half_thresh_diff)(vm_z_log_var)

    layer = tf.stack([gauss_z_mean,gauss_z_log_var])
    gauss = tfpl.DistributionLambda(make_distribution_fn = lambda t: tfd.MultivariateNormalDiag(t[0],tf.exp(t[1]/2)),
                                    name="gauss_distribution")(layer)
    
    layer = tf.stack([vm_z_mean,vm_z_log_var])
    vonmis = tfpl.DistributionLambda(make_distribution_fn = lambda t: tfd.VonMises(t[0],tf.exp(-t[1])),
                                     name="vm_distribution")(layer)
    
    centers = tfkl.Concatenate()([gauss_z_mean, vm_z_mean])
    log_vars = tfkl.Concatenate()([gauss_z_log_var, vm_z_log_var])
    samples = tfkl.Concatenate()([gauss,vonmis])
    
    encoder = tf.keras.Model(data_inputs,[centers,log_vars,samples])

    return encoder, [gauss, vonmis]

def make_decoder_oneparticle(latent_dim = 1,
                             dense_sizes = [64,64,64],
                             real_dim = 2):
    
    latent_inputs = tfk.Input(shape=(latent_dim,), name='z_sampling')
    layer = latent_inputs

    for layer_size in dense_sizes:
        layer = tfkl.Dense(layer_size,activation='relu')(layer)

    outputs = tfkl.Dense(real_dim, name='outputs', activation = None)(layer)
    
    decoder = tf.keras.Model(latent_inputs,outputs)

    return decoder

def make_vae_oneparticle(real_dim = 2,
                         latent_dims_line = 1,
                         latent_dims_circle = 1,
                         optimizer = tfk.optimizers.Adam(lr=0.001,clipnorm=0.1),
                         loss_type = "2D"):
    
    latent_dim = latent_dims_line + latent_dims_circle
    
    data_inputs = tfk.Input(shape=(real_dim,))
    beta_inputs = tfk.Input(shape=(1,))
    
    encoder, distributions = make_encoder_oneparticle(data_inputs,
                                                      latent_dims_line = latent_dims_line,
                                                      latent_dims_circle = latent_dims_circle,
                                                      real_dim = real_dim,
                                                      dense_sizes = [64,64,64])
    
    gauss, vonmis = distributions
    
    decoder = make_decoder_oneparticle(latent_dim = latent_dim,
                                       dense_sizes = [64,64,64],
                                       real_dim = real_dim)
    
    sample = encoder(data_inputs)[-1]
    
    vae = tf.keras.Model([data_inputs,beta_inputs],decoder(sample))
    
    
    def beta_vae_loss(beta):
        def vae_loss(x, x_decoded_mean):
            
            if loss_type is "circle":
                diff = (x_decoded_mean - x)
                diffmod = tf.math.mod(diff + np.pi, 2*np.pi) - np.pi
                recon_loss = tf.reduce_sum(tf.square(diffmod),axis=-1)
            else:
                recon_loss = tf.reduce_sum(tf.square(x - x_decoded_mean),axis=-1)
            
            zeros = tf.fill(tf.shape(vonmis),0.)
            uniform = tfd.VonMises(zeros,zeros)
            kl_loss_vm = tf.reduce_sum(vonmis.kl_divergence(other=uniform),axis=-1)
            
            zeros = tf.fill(tf.shape(gauss),0.)
            ones = tf.fill(tf.shape(gauss),1.)
            standard_normal = tfd.MultivariateNormalDiag(loc=zeros,scale_diag=ones)
            kl_loss_gauss = gauss.kl_divergence(other=standard_normal)

            return tf.reduce_mean(recon_loss/beta**2 + kl_loss_vm + kl_loss_gauss)
        return vae_loss
    
    def recon_loss(x, x_decoded_mean):
#         if loss_type is "circle":
#             recon_loss = tf.reduce_sum(tf.square(tf.acos(tf.cos(x - x_decoded_mean))),axis=-1)
        if loss_type is "circle":
            diff = (x_decoded_mean - x)
            diffmod = tf.math.mod(diff + np.pi, 2*np.pi) - np.pi
            recon_loss = tf.reduce_sum(tf.square(diffmod),axis=-1)
        else:
            recon_loss = tf.reduce_sum(tf.square(x - x_decoded_mean),axis=-1)
        return tf.reduce_mean(recon_loss,axis=-1)
    

    def kl_loss(x, x_decoded_mean):
        
        zeros = tf.fill(tf.shape(vonmis),0.)
        uniform = tfd.VonMises(zeros,zeros)
        kl_loss_vm = tf.reduce_sum(vonmis.kl_divergence(other=uniform),axis=-1)

        zeros = tf.fill(tf.shape(gauss),0.)
        ones = tf.fill(tf.shape(gauss),1.)
        standard_normal = tfd.MultivariateNormalDiag(loc=zeros,scale_diag=ones)
        kl_loss_gauss = gauss.kl_divergence(other=standard_normal)
            
        return tf.reduce_mean( kl_loss_vm + kl_loss_gauss)

    
    vae.compile(loss=beta_vae_loss(beta_inputs),
                optimizer=optimizer,
                experimental_run_tf_function=False,
                metrics = [recon_loss,kl_loss]
               )
    
    vae.summary()
    
    return vae, encoder, decoder