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
                             real_dim=2,
                             output_vm_log_var = False,
                             train_batchnorm = True,
                            precision=tf.float32):
    
    layer = data_inputs
    
    
    for i, layer_size in enumerate(dense_sizes):
        layer = tfkl.Dense(layer_size,
                           activation='relu',
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform',
                           name = 'encoder_dense_' + str(i)#,
                          #dtype=precision
                          )(layer)
    
    gauss_z_mean = tfkl.Dense(latent_dims_line, name='encoder_gauss_z_mean',activation=None#,
                             #dtype=precision
                             )(layer)
    gauss_z_log_var = tfkl.Dense(latent_dims_line, name='encoder_gauss_z_log_var',activation=None#,
                                #dtype=precision
                                )(layer)
    
    vm_z_mean_x = tfkl.Dense(latent_dims_circle, name='encoder_vm_z_mean_x',activation=None#,
#                            kernel_initializer = tf.random_normal_initializer(stddev=10.),
                            #dtype=precision
                            )(layer)
#     vm_z_mean_x = tfkl.BatchNormalization(renorm=True,
#                                           momentum=0.9999,
#                                           renorm_momentum=0.9999,
#                                           trainable = train_batchnorm,
#                                           renorm_clipping={'rmax':3.,'rmin':1./3,'dmax':5.},
#                                           name='encoder_vm_z_mean_x_batchnorm'
#                                          )(vm_z_mean_x)

    vm_z_mean_y = tfkl.Dense(latent_dims_circle, name='encoder_vm_z_mean_y',activation=None,
#                            kernel_initializer = tf.random_normal_initializer(stddev=10.),
                            #dtype=precision
                            )(layer)
#     vm_z_mean_y = tfkl.BatchNormalization(renorm=True,
#                                           momentum=0.9999,
#                                           renorm_momentum=0.9999,
#                                           trainable = train_batchnorm,
#                                           renorm_clipping={'rmax':3.,'rmin':1./3,'dmax':5.},
#                                           name='encoder_vm_z_mean_y_batchnorm'
#                                          )(vm_z_mean_y)
    vm_z_mean = tf.atan2(vm_z_mean_x,vm_z_mean_y, name = 'encoder_vm_z_mean')

#     vm_z_mean = tfkl.Dense(latent_dims_circle, name='vm_z_mean',activation=None)(layer)
#     vm_z_mean = tfkl.Lambda(lambda x:  x*100)(vm_z_mean)
    
    #exp threshold
#     up_thresh = 1.0
#     low_thresh = -30
#     half_thresh_diff = (up_thresh - low_thresh)/2
#     cprime = np.arctanh((half_thresh_diff-up_thresh)/half_thresh_diff)
    vm_z_log_var = tfkl.Dense(latent_dims_circle, name='encoder_vm_z_log_var',activation=None#,dtype=precision
                             )(layer)
#     vm_z_log_var = tfkl.Lambda(lambda x:  x/half_thresh_diff+cprime)(vm_z_log_var)
#     vm_z_log_var = tfkl.Activation('tanh')(vm_z_log_var)
#     vm_z_log_var = tfkl.Lambda(lambda x:  x*half_thresh_diff + up_thresh - half_thresh_diff)(vm_z_log_var)

    layer = tf.stack([gauss_z_mean,gauss_z_log_var])
    gauss = tfpl.DistributionLambda(make_distribution_fn = lambda t: tfd.MultivariateNormalDiag(t[0],tf.exp(t[1]/2)),
                                    name="encoder_gauss_distribution"#,dtype=precision
                                   )(layer)
    
    layer = tf.stack([vm_z_mean,vm_z_log_var])
    vonmis = tfpl.DistributionLambda(make_distribution_fn = lambda t: tfd.VonMises(t[0],tf.where(t[1]<0.,tf.exp(-t[1])-1.,0.)),
                                     name="encoder_vm_distribution"#,dtype=precision
                                    )(layer)
    
    centers = tfkl.Concatenate()([gauss_z_mean, vm_z_mean])
    log_vars = tfkl.Concatenate()([gauss_z_log_var, vm_z_log_var])
    samples = tfkl.Concatenate()([gauss,vonmis])
    
    encoder = tf.keras.Model(data_inputs,[centers,log_vars,samples])
    
                                     
    if output_vm_log_var:
        return encoder, [gauss, vonmis], vm_z_log_var
    else:
        return encoder, [gauss, vonmis]

def make_decoder_oneparticle(latent_dims_line = 1,
                             latent_dims_circle = 1,
                             dense_sizes = [64,64,64],
                             real_dim = 2,
                            precision=tf.float32):
    
    latent_inputs = tfk.Input(shape=(latent_dims_line + latent_dims_circle,), name='z_sampling'#,dtype=precision
                             )
    line_dims = latent_inputs[:,:latent_dims_line]
    circle_dims = latent_inputs[:,latent_dims_line:]
    
    circle_x = tf.sin(circle_dims)
    circle_y = tf.cos(circle_dims)
    
    layer = tfkl.Concatenate()([line_dims,circle_x,circle_y])

    for i, layer_size in enumerate(dense_sizes):
        layer = tfkl.Dense(layer_size,activation='relu',name = 'decoder_dense_' + str(i)#,
                          #dtype=precision
                          )(layer)

    outputs = tfkl.Dense(real_dim, name='outputs', activation = None#,
                        #dtype=precision
                        )(layer)
    
    decoder = tf.keras.Model(latent_inputs,outputs)

    return decoder

def make_vae_oneparticle(real_dim = 2,
                         latent_dims_line = 1,
                         latent_dims_circle = 1,
                         optimizer = tfk.optimizers.Adam(lr=0.001,clipnorm=0.1),
                         dense_sizes = [64,64,64],
                         loss_type = "2D",
                         train_batchnorm = True,
                        precision=tf.float32):
    
    latent_dim = latent_dims_line + latent_dims_circle
    
    data_inputs = tfk.Input(shape=(real_dim,)#,dtype='float64'
                           )
    beta_inputs = tfk.Input(shape=(1,)#,dtype='float64'
                           )
    alpha_inputs = tfk.Input(shape=(1,)#,dtype='float64'
                            )
    
    encoder, distributions, vm_log_var  = make_encoder_oneparticle(data_inputs,
                                                      latent_dims_line = latent_dims_line,
                                                      latent_dims_circle = latent_dims_circle,
                                                      real_dim = real_dim,
                                                      dense_sizes = dense_sizes,
                                                      output_vm_log_var = True,
                                                      train_batchnorm = train_batchnorm,
                                                      precision=precision)
    
    gauss, vonmis = distributions
    
    decoder = make_decoder_oneparticle(latent_dims_line = latent_dims_line,
                                       latent_dims_circle = latent_dims_circle,
                                       dense_sizes = dense_sizes,
                                       real_dim = real_dim,
                                       precision=precision)
    
    sample = encoder(data_inputs)[-1]
    
    vae = tf.keras.Model([data_inputs,beta_inputs,alpha_inputs],decoder(sample))
    
    concentration1 = tf.convert_to_tensor(vonmis.concentration)
    #concentration1 = tf.cast(concentration1,tf.float64)

    i0e_concentration1 = tf.math.bessel_i0e(concentration1)
    i1e_concentration1 = tf.math.bessel_i1e(concentration1)
    kl_loss_vm =  tf.reduce_sum((-concentration1 + tf.math.log(1 / i0e_concentration1) + concentration1 * (i1e_concentration1 / i0e_concentration1)),axis=-1)
#     kl_loss_vm = tf.cast(kl_loss_vm,tf.float32)
    

    
    def beta_vae_loss(beta, alpha):
#         alpha = tf.cast(alpha,tf.float64)
#         beta = tf.cast(beta,tf.float64)
        def vae_loss(x, x_decoded_mean):
            

            
#             x = tf.cast(x,tf.float64)
#             x_decoded_mean = tf.cast(x_decoded_mean,tf.float64)
            
            if loss_type is "circle":
                diff = (x_decoded_mean - x)
                diffmod = tf.math.mod(diff + np.pi, 2*np.pi) - np.pi
                recon_loss = tf.reduce_sum(tf.square(diffmod),axis=-1)
            else:
                recon_loss = tf.reduce_sum(tf.square(x - x_decoded_mean),axis=-1)
            
#             zeros = tf.fill(tf.shape(vonmis),0.)
#             uniform = tfd.VonMises(zeros,zeros)
#             kl_loss_vm = tf.reduce_sum(vonmis.kl_divergence(other=uniform),axis=-1)
            
#             zeros = tf.cast(tf.fill(tf.shape(gauss),0.),tf.float64)
            #ones = tf.cast(tf.fill(tf.shape(gauss),1.),tf.float64)
            ones = tf.keras.backend.ones(tf.shape(gauss))
            zeros = tf.keras.backend.zeros(tf.shape(gauss))
#             zeros = tf.fill(tf.shape(gauss),0.)
#             ones = tf.fill(tf.shape(gauss),1.)
            standard_normal = tfd.MultivariateNormalDiag(loc=zeros,scale_diag=ones)
            kl_loss_gauss = gauss.kl_divergence(other=standard_normal)
            
            extra_term = tf.reduce_sum(tf.where(vm_log_var>0,tf.exp(vm_log_var),0.),axis=-1)

            return tf.reduce_mean(recon_loss/beta**2 + kl_loss_vm*alpha + kl_loss_gauss+extra_term)
        return vae_loss
    

    
    def recon_loss(x, x_decoded_mean):
#         x = tf.cast(x,tf.float64)
           
#         x_decoded_mean = tf.cast(x_decoded_mean,tf.float64)
        
#         if loss_type is "circle":
#             recon_loss = tf.reduce_sum(tf.square(tf.acos(tf.cos(x - x_decoded_mean))),axis=-1)
        if loss_type is "circle":
            diff = (x_decoded_mean - x)
            diffmod = tf.math.mod(diff + np.pi, 2*np.pi) - np.pi
            recon_loss = tf.reduce_sum(tf.square(diffmod),axis=-1)
        else:
            recon_loss = tf.reduce_sum(tf.square(x - x_decoded_mean),axis=-1)
        return tf.reduce_mean(recon_loss,axis=-1)
    

    def kl_loss_gauss(x, x_decoded_mean):
#         x = tf.cast(x,tf.float64)
#         x_decoded_mean = tf.cast(x_decoded_mean,tf.float64)
#         zeros = tf.fill(myshapetens,0.)
#         uniform = tfd.VonMises(zeros,zeros)
#         kl_loss_vm = tf.reduce_sum(vonmis.kl_divergence(other=uniform),axis=-1)

#         zeros = tf.cast(tf.fill(tf.shape(gauss),0.),tf.float64)
#         ones = tf.cast(tf.fill(tf.shape(gauss),1.),tf.float64)
        ones = tf.keras.backend.ones(tf.shape(gauss))
        zeros = tf.keras.backend.zeros(tf.shape(gauss))
#         zeros = tf.fill(tf.shape(gauss),0.)
#         ones = tf.fill(tf.shape(gauss),1.)
        standard_normal = tfd.MultivariateNormalDiag(loc=zeros,scale_diag=ones)
        kl_loss_gauss = gauss.kl_divergence(other=standard_normal)
            
        return tf.reduce_mean(kl_loss_gauss,axis=-1)

    def kl_loss_VM(x, x_decoded_mean):
        return tf.reduce_mean(kl_loss_vm,axis=-1)
    
    vae.compile(loss=beta_vae_loss(beta_inputs, alpha_inputs),
                optimizer=optimizer,
                experimental_run_tf_function=False,
                metrics = [recon_loss,kl_loss_gauss,kl_loss_VM]
               )
    
    vae.summary()
    
    
    return vae, encoder, decoder