import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten, Reshape, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from utils.tf_sinkhorn import ground_distance_tf_nograd, sinkhorn_knopp_tf_scaling_stabilized_class
from tensorflow.python.keras.utils import tf_utils


import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

import numpy as np



loss_tracker = keras.metrics.Mean(name="loss")
recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
KL_loss_tracker = keras.metrics.Mean(name="KL_loss")


class betaVAEModel(keras.Model):

    def betaVAE_compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                recon_loss=None,
                KL_loss=None,
                encoder = None,
                decoder = None,
                **kwargs):

        self.compile(optimizer=optimizer,
                    #loss=loss,
                    metrics=metrics,
                    loss_weights=loss_weights,
                    sample_weight_mode=sample_weight_mode,
                    weighted_metrics=weighted_metrics,
                    **kwargs)
        self.recon_loss = recon_loss
        self.KL_loss = KL_loss
        self.beta_r = tf.Variable(1.,trainable=False, name="beta_r")
        self.encoder = encoder
        self.decoder = decoder

    @tf.function
    def train_step(self, data):
        xpair, y = data
        x,logbeta = xpair

        with tf.GradientTape() as tape:
            y_pred ,z_mean, z_log_var, z = self(xpair, training=True)  # Forward pass
            # Compute our own loss
            recon_loss = self.recon_loss(y, y_pred)
            KL_loss = self.KL_loss(z_mean, z_log_var)

            loss = tf.reduce_mean(recon_loss) + tf.reduce_mean(tf.exp(logbeta*np.log(10.))*self.beta_r * KL_loss)


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        recon_loss_tracker.update_state(tf.reduce_mean(recon_loss))
        KL_loss_tracker.update_state(tf.reduce_mean(KL_loss))

        return {"loss": loss_tracker.result(),
                "recon_loss": recon_loss_tracker.result(),
                "KL loss": KL_loss_tracker.result()}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        xpair, y = data
        x,logbeta = xpair

        # Compute predictions
        y_pred ,z_mean, z_log_var, z = self(xpair, training=False)  # Forward pass
        # Compute our own loss
        recon_loss = self.recon_loss(y, y_pred)
        KL_loss = self.KL_loss(z_mean, z_log_var)

        loss_tracker.reset_states()
        recon_loss_tracker.reset_states()
        KL_loss_tracker.reset_states()
        
        loss_tracker.update_state(tf.reduce_mean(recon_loss + tf.exp(logbeta*np.log(10.))*self.beta_r*KL_loss))
        recon_loss_tracker.update_state(tf.reduce_mean(recon_loss))
        KL_loss_tracker.update_state(tf.reduce_mean(KL_loss))

        return {"loss": loss_tracker.result(),
                "recon_loss": recon_loss_tracker.result(),
                "KL loss": KL_loss_tracker.result()}
    
    def predict_mle(self,data):
        _,logbeta = data
        z_mean, z_log_var, z = self.encoder(data, training=False)
        return self.decoder([z_mean, logbeta])


    @tf.function
    def heat_capacity_D(self,data,nsamples=1):
        xpair, y = data
        x, logbeta = xpair
        beta = tf.exp(logbeta*np.log(10.))
        y_pred ,z_mean, z_log_var, z = self([tf.tile(x,[nsamples,1,1]), tf.math.log(tf.tile(beta,[nsamples]))/np.log(10.)], training=False)  # Forward pass
        recon_loss = self.recon_loss(tf.tile(y,[nsamples,1,1]), y_pred)/nsamples
        #KL_loss = self.KL_loss(z_mean, z_log_var)

        return tf.gradients(recon_loss, beta)[0]#/tf.exp(logbeta), tf.gradients(KL_loss, -logbeta)

    @tf.function
    def heat_capacity_D_KL(self,data,nsamples=1):
        xpair, y = data
        x, logbeta = xpair
        y_pred ,z_mean, z_log_var, z = self([tf.tile(x,[nsamples,1,1]), tf.tile(logbeta,[nsamples])], training=False)  # Forward pass
        recon_loss = tf.reduce_mean(self.recon_loss(tf.tile(y,[nsamples,1,1]), y_pred))
        KL_loss = self.KL_loss(z_mean, z_log_var)

        return tf.gradients(recon_loss, logbeta)[0]/tf.exp(logbeta), tf.gradients(-KL_loss, logbeta)[0]/np.log(10.)

    @tf.function
    def heat_capacity_KL(self,data):
        xpair, y = data
        x, logbeta = xpair
        y_pred ,z_mean, z_log_var, z = self([x,logbeta], training=False)  # Forward pass
        KL_loss = self.KL_loss(z_mean, z_log_var)

        return tf.gradients(-KL_loss, logbeta)[0]/np.log(10.)

# https://arxiv.org/pdf/1611.00712.pdf

def build_and_compile_annealing_vae(encoder_conv_layers = [256,256,256,256],
                                    dense_size = [256,256,256,256],
                                    decoder_sizes = [512,256,256,256],
                                    verbose=0,dropout=0,
                                    latent_dim = 128,
                                    optimizer=keras.optimizers.Adam(),
                                    num_particles_out = 50,
                                    reg_init = 1.,
                                    reg_final = 0.01,
                                    numItermaxinner = 10,
                                    numIter = 10,
                                    stopThr=1e-3,
                                    temp = 0.3,
                                    EPSILON = 1e-6,
                                    num_particles_in = 100,
                                    check_err_period = 10,
                                    num_inputs = 4,
                                    exponent = 1,
                                    mod2pi=True,
                                    **kwargs):

  
    #Encoder
    inputs = tf.keras.Input(shape=(num_particles_in,num_inputs,), name='inputs')
    logbeta = tf.keras.Input(shape=(1,), name='logbeta_input')
    layer = inputs

    for layer_size in encoder_conv_layers:
        layer = Conv1D(layer_size,1)(layer)
        layer = keras.layers.LeakyReLU(0.1)(layer)
        if dropout > 0:
            layer = keras.layers.Dropout(dropout,noise_shape=(None,1,layer_size))(layer)
    
    # Sum layer
    layer = tf.keras.backend.sum(layer,axis=1)/np.sqrt(encoder_conv_layers[-1])
    layer = tf.keras.layers.Concatenate()([layer,logbeta])

    # Dense layers
    for size in dense_size:
        layer = Dense(size)(layer)
        layer = keras.layers.LeakyReLU(0.1)(layer)
        if dropout > 0:
            layer = keras.layers.Dropout(dropout)(layer)
     
    z_mean = Dense(latent_dim, name='z_mean')(layer)
    z_log_var = Dense(latent_dim, name='z_log_var')(layer)
    
    
    layer = tf.stack([z_mean,z_log_var])
    z = tfpl.DistributionLambda(make_distribution_fn = lambda t: tfd.MultivariateNormalDiag(t[0],tf.exp(t[1]/2)),
                                    name="encoder_gauss_distribution"#,dtype=precision
                                   )(layer)
    
#     z = Lambda(sampling_gauss, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model([inputs,logbeta], [z_mean, z_log_var, z], name='encoder')
    if verbose:
        encoder.summary()
    #plot_model(encoder, to_file='CNN-VAE_encoder.png', show_shapes=True)

    # Decoder
    latent_inputs_gauss = Input(shape=(latent_dim,), name='z_sampling')
    logbeta_decoder = Input(shape=(1,), name='logbeta_decoder')
    layer = tf.keras.layers.Concatenate()([latent_inputs_gauss,logbeta_decoder])

    #layer = latent_inputs
    
    for i, layer_size in enumerate(decoder_sizes):
        layer = Dense(layer_size)(layer)
        layer = keras.layers.LeakyReLU(0.1)(layer)
        if dropout > 0:
            layer = keras.layers.Dropout(dropout)(layer)

    if mod2pi:
        numnodes = 4
    else:
        numnodes = 3
    layer = Dense(num_particles_out*numnodes)(layer)
    layer = Reshape((num_particles_out,numnodes))(layer)
    layer_pT = layer[:,:,0:1]
    layer_pT = tf.keras.layers.Softmax(axis=-2)(layer_pT)
    layer_eta = layer[:,:,1:2]
    if mod2pi:
        layer_phi = tf.math.atan2(layer[:,:,3],layer[:,:,2])
        layer_phi = tf.expand_dims(layer_phi,axis=-1)
    else:
        layer_phi = layer[:,:,2:3]
    decoded = tf.keras.layers.Concatenate()([layer_pT,layer_eta,layer_phi])

    decoder = Model([latent_inputs_gauss, logbeta_decoder], decoded, name='decoder')
    if verbose:
        decoder.summary()
    #plot_model(decoder, to_file='CNN-VAE_decoder.png', show_shapes=True)


    outputs = decoder([encoder([inputs,logbeta])[2],logbeta])
    vae = betaVAEModel([inputs,logbeta], [outputs,z_mean, z_log_var, z], name='VAE')


    sinkhorn_knopp_tf_inst = sinkhorn_knopp_tf_scaling_stabilized_class(reg_init,
                                                                            reg_final,
                                                                            numItermaxinner=numItermaxinner,
                                                                            numIter=numIter,
                                                                            stopThr=stopThr,
                                                                            check_err_period = check_err_period,
                                                                            dtype=tf.float64,
                                                                            sparse = False)   

  
    def return_return_loss(pt_outs, x_outs, pt_in, x_in):
        @tf.custom_gradient
        def return_loss(pt_out, x_out):

            epsilon = np.float64(1e-10)
            ground_distance = tf.pow(ground_distance_tf_nograd(x_in,x_out,mod2pi=mod2pi),exponent)

            match = sinkhorn_knopp_tf_inst(pt_in, pt_out, tf.stop_gradient(ground_distance))        
            recon_loss = tf.linalg.trace(tf.matmul(tf.stop_gradient(tf.cast(match,tf.float32)),ground_distance,transpose_b=True))
            
            def grad(dL):
                aones = tf.fill(tf.shape(pt_in),np.float64(1.))
                bones = tf.fill(tf.shape(pt_out),np.float64(1.))

                Mnew = tf.cast(tf.transpose(ground_distance,perm=[0,2,1]),tf.float64)

                T = tf.cast(tf.transpose(match,perm=[0,2,1]),tf.float64)
                Ttilde = T[:,:,:-1]

                L = T * Mnew
                Ltilde = L[:,:,:-1]

                D1 = tf.linalg.diag(tf.reduce_sum(T,axis=-1))
                D2 = tf.linalg.diag(1/(tf.reduce_sum(Ttilde,axis=-2) + np.float64(1e-100))) # Add epsilon to ensure invertibility

                H = D1 - tf.matmul(tf.matmul(Ttilde,D2),Ttilde,transpose_b=True) + epsilon* tf.eye(num_rows = tf.shape(bones)[-1],batch_shape = [tf.shape(bones)[0]],dtype=tf.float64) # Add small diagonal piece to make sure H is invertible in edge cases.

                f = - tf.reduce_sum(L,axis=-1) + tf.squeeze(tf.matmul(tf.matmul(Ttilde,D2),tf.expand_dims(tf.reduce_sum(Ltilde,axis=-2),-1)),axis=-1)
                g = tf.squeeze(tf.matmul(tf.linalg.inv(H),tf.expand_dims(f,-1)),axis=-1)

                grad_pT = g - bones*tf.expand_dims(tf.reduce_sum(g,axis=-1),-1)/tf.cast(tf.shape(bones)[1],tf.float64)
                
                grad_x_out = tf.gradients(recon_loss,x_out)[0]
                
                return [-tf.expand_dims(dL,-1) * tf.cast(grad_pT,tf.float32),
                        tf.expand_dims(tf.expand_dims(dL,-1),-1)*tf.cast(grad_x_out,tf.float32)]

            return recon_loss, grad
        return return_loss(pt_outs, x_outs)


    @tf.function
    def recon_loss(x, x_decoded_mean):
        pt_out = x_decoded_mean[:,:,0]
        x_out = x_decoded_mean[:,:,1:]
        pt_in = x[:,:,0]
        x_in = x[:,:,1:]
        return return_return_loss(pt_out, x_out, pt_in, x_in)

    @tf.function
    def kl_loss(z_mean, z_log_var):
        return -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)


    vae.betaVAE_compile(recon_loss=recon_loss,
                        KL_loss = kl_loss,
                        optimizer=optimizer,experimental_run_tf_function=False,
                        encoder = encoder, decoder = decoder
                        #metrics = [recon_loss,kl_loss(beta_input), kl_loss_bern(beta_input)]
                       )
    
    vae.summary()
    
    return vae, encoder, decoder

class reset_metrics(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        loss_tracker.reset_states()
        recon_loss_tracker.reset_states()
        KL_loss_tracker.reset_states()
#         val_loss_tracker.reset_states()
#         val_recon_loss_tracker.reset_states()
#         val_KL_loss_tracker.reset_states()

class myTerminateOnNaN(keras.callbacks.Callback):
  """Callback that terminates training when a NaN loss is encountered.
  """

  def __init__(self):
    super(myTerminateOnNaN, self).__init__()
    self._supports_tf_logs = True

  def on_epoch_end(self, batch, logs=None):
    logs = logs or {}
    loss = logs.get('loss')
    if loss is not None:
      loss = tf_utils.to_numpy_or_python_type(loss)
      if np.isnan(loss) or np.isinf(loss):
        print('Batch %d: Invalid loss, terminating training' % (batch))
        self.model.stop_training = True
