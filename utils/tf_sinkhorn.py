from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import ot as pot
import math
import tensorflow_probability as tfp

def greenkhorn_tf(a, b, M, reg, adaptive_min=None, numItermax=10000, stopThr=1e-9, log=False):
    M = tf.cast(M, tf.float64)
    a = tf.cast(a, tf.float64)
    b = tf.cast(b, tf.float64)
    
    if reg == 'adaptive':
        maxd = tf.reduce_max(M,axis=[-1,-2])
        if adaptive_min is None:
            reg = maxd/np.float64(708.)
        else:
            reg = tf.maximum(maxd/np.float64(708.), np.float64(adaptive_min))
        K = tf.exp(-M/tf.expand_dims(tf.expand_dims(reg,-1),-1))
    else:
        K = tf.exp(-M/reg)
        
    u = tf.fill(tf.shape(a),np.float64(1.))/tf.cast(tf.shape(a)[1],tf.float64)
    v = tf.fill(tf.shape(b),np.float64(1.))/tf.cast(tf.shape(b)[1],tf.float64)
    G = tf.expand_dims(u,-1)*K*tf.expand_dims(v,0)
    
    viol = tf.reduce_sum(G,axis = 2) - a
    viol_2 = tf.reduce_sum(G,axis = 1) - b
    stopThr_val = 1
    
    if log:
        log = dict()
        log['u'] = u
        log['v'] = v
    
    def loopfn():
        i_1 = tf.argmax(tf.abs(viol))
        i_2 = tf.argmax(tf.abs(viol_2))
        m_viol_1 = tf.abs(viol[i_1])
    

# def ground_distance_tf(pointsa,pointsb,gradients=False):
    
#     a_dim = tf.shape(pointsa)[-2]
#     b_dim = tf.shape(pointsb)[-2]
    
#     amat = tf.tile(tf.expand_dims(pointsa,2),[1,1,b_dim,1])
#     bmat = tf.tile(tf.expand_dims(pointsb,1),[1,a_dim,1,1])
    
#     if gradients:
#         return (bmat - amat) / tf.norm(bmat - amat,axis=3)
#     else:
#         return tf.norm(bmat - amat,axis=3)


def sinkhorn_knopp_tf(a, b, M, reg, adaptive_min=None, numItermax=1000, stopThr=1e-9, verbose=False, **kwargs):
    M = tf.cast(M, tf.float64)
    a = tf.cast(a, tf.float64)
    b = tf.cast(b, tf.float64)
    if reg == 'adaptive':
        maxd = tf.reduce_max(M,axis=[-1,-2])
        if adaptive_min is None:
            reg = maxd/np.float64(708.)
        else:
            reg = tf.maximum(maxd/np.float64(708.), np.float64(adaptive_min))
            #reg = tf.maximum(maxd/np.float64(300.), np.float64(adaptive_min))
        K = tf.exp(-M/tf.expand_dims(tf.expand_dims(reg,-1),-1))
    else:
        K = tf.exp(-M/reg)
    #u = tf.ones(a.shape)/tf.reduce_sum(a,axis=-1,keepdims=True)
    u = tf.fill(tf.shape(a),np.float64(1.))/tf.cast(tf.shape(a)[1],tf.float64)
    v = tf.fill(tf.shape(b),np.float64(1.))/tf.cast(tf.shape(b)[1],tf.float64)
#     uprev = tf.fill(tf.shape(a),np.float64(1.))/tf.reduce_sum(a,axis=-1,keepdims=True)
#     vprev = tf.fill(tf.shape(b),np.float64(1.))/tf.reduce_sum(b,axis=-1,keepdims=True)
    uprev = tf.fill(tf.shape(a),np.float64(1.))
    vprev = tf.fill(tf.shape(b),np.float64(1.))
    Kp = tf.expand_dims(1/a,-1)*K
    
    err = tf.Variable(1.,dtype=tf.float64)
    cpt = tf.Variable(0)
    
    flag = tf.Variable(1)
    
    mycond = lambda flag, err, cpt, Kp, u, v, uprev, vprev : tf.logical_and(tf.less(cpt, numItermax),tf.greater(err,stopThr))


    def loopfn(flag, err, cpt, Kp, u, v, uprev, vprev):
        uprev = u
        vprev = v
        
        KtransposeU = tf.squeeze(tf.matmul(K,tf.expand_dims(u,-1),transpose_a=True),axis=-1)
        v = b / KtransposeU
        #u = 1/tf.squeeze(tf.matmul(Kp, tf.expand_dims(v,-1)),axis=-1)
        u = a/tf.squeeze(tf.matmul(K, tf.expand_dims(v,-1)),axis=-1)
        
#         error_cond = tf.reduce_any(tf.equal(KtransposeU,0))

        error_cond = tf.reduce_any(tf.math.is_nan(u))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_nan(v)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_inf(u)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_inf(v)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.equal(KtransposeU,0)))
        
        def error_function_true():
            return tf.Variable(numItermax), uprev, vprev
        def error_function_false():
            return cpt+1, u, v
        cpt, u, v = tf.cond(error_cond,error_function_true,error_function_false)

#         def cptmod10_true():
            
#             tmp2 = tf.squeeze(tf.matmul(tf.expand_dims(u,-2),K),axis=1)*v
#             #tmp2 = tf.einsum('ai,aij,aj->aj', u, K, v)
#             newerr = tf.norm(tmp2-b,axis=-1)
#             stopthr_cond = tf.reduce_all(tf.less(newerr,stopThr))
            
#             def stopthr_false():
#                 return tf.reduce_max(newerr), flag + 1, cpt
#             def stopthr_true():
#                 return tf.reduce_max(newerr), flag + 1, tf.Variable(numItermax)
            
#             return tf.cond(stopthr_cond,stopthr_true,stopthr_false)
        
#         def cptmod10_false():
#             return err, flag, cpt
        
#         cptmod10_cond = tf.equal(tf.floormod(cpt,10),0)
#         err, flag, cpt = tf.cond(cptmod10_cond,cptmod10_true,cptmod10_false)
        
        return flag, err, cpt, Kp, u, v, uprev, vprev
    
    this = tf.while_loop(mycond, loopfn,[flag, err, cpt,  Kp, u, v, uprev, vprev])

    u = this[4]
    v = this[5]
    
    return tf.cast(tf.expand_dims(u,-1)*K*tf.expand_dims(v,-2), tf.float32)


def sinkhorn_knopp_tf_64(a, b, M, reg, adaptive_min=None, numItermax=1000, stopThr=1e-9, verbose=False, **kwargs):
    M = tf.cast(M, tf.float64)
    a = tf.cast(a, tf.float64)
    b = tf.cast(b, tf.float64)
    if reg == 'adaptive':
        maxd = tf.reduce_max(M,axis=[-1,-2])
        if adaptive_min is None:
            reg = maxd/np.float64(708.)
        else:
            reg = tf.maximum(maxd/np.float64(708.), np.float64(adaptive_min))
            #reg = tf.maximum(maxd/np.float64(300.), np.float64(adaptive_min))
        K = tf.exp(-M/tf.expand_dims(tf.expand_dims(reg,-1),-1))
    else:
        K = tf.exp(-M/reg)
    #u = tf.ones(a.shape)/tf.reduce_sum(a,axis=-1,keepdims=True)
    u = tf.fill(tf.shape(a),np.float64(1.))/tf.cast(tf.shape(a)[1],tf.float64)
    v = tf.fill(tf.shape(b),np.float64(1.))/tf.cast(tf.shape(b)[1],tf.float64)
#     uprev = tf.fill(tf.shape(a),np.float64(1.))/tf.reduce_sum(a,axis=-1,keepdims=True)
#     vprev = tf.fill(tf.shape(b),np.float64(1.))/tf.reduce_sum(b,axis=-1,keepdims=True)
    uprev = tf.fill(tf.shape(a),np.float64(1.))
    vprev = tf.fill(tf.shape(b),np.float64(1.))
    Kp = tf.expand_dims(1/a,-1)*K

    err = tf.Variable(1.,dtype=tf.float64,trainable=False)
    cpt = tf.Variable(0,trainable=False)


    flag = tf.Variable(1,trainable=False)


    mycond = lambda flag, err, cpt, Kp, u, v, uprev, vprev : tf.logical_and(tf.less(cpt, numItermax),tf.greater(err,stopThr))


    def loopfn(flag, err, cpt, Kp, u, v, uprev, vprev):
        uprev = u
        vprev = v
        
        KtransposeU = tf.squeeze(tf.matmul(K,tf.expand_dims(u,-1),transpose_a=True),axis=-1)
        v = b / KtransposeU
        #u = 1/tf.squeeze(tf.matmul(Kp, tf.expand_dims(v,-1)),axis=-1)
        u = a/tf.squeeze(tf.matmul(K, tf.expand_dims(v,-1)),axis=-1)
        
#         error_cond = tf.reduce_any(tf.equal(KtransposeU,0))

        error_cond = tf.reduce_any(tf.math.is_nan(u))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_nan(v)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_inf(u)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_inf(v)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.equal(KtransposeU,0)))
        
        def error_function_true():
            #return tf.Variable(numItermax,trainable=False), uprev, vprev
            return tf.Variable(numItermax,trainable=False,name='bob'), uprev, vprev
        def error_function_false():
            return cpt+1, u, v
        cpt, u, v = tf.cond(error_cond,error_function_true,error_function_false)

        def cptmod10_true():
            
            tmp2 = tf.squeeze(tf.matmul(tf.expand_dims(u,-2),K),axis=1)*v
            #tmp2 = tf.einsum('ai,aij,aj->aj', u, K, v)
            newerr = tf.norm(tmp2-b,axis=-1)
            stopthr_cond = tf.reduce_all(tf.less(newerr,stopThr))
            
            def stopthr_false():
                return tf.reduce_max(newerr), flag + 1, cpt
            def stopthr_true():
                return tf.reduce_max(newerr), flag + 1, tf.Variable(numItermax,trainable=False,name='Alice')
            
            return tf.cond(stopthr_cond,stopthr_true,stopthr_false)
        
        def cptmod10_false():
            return err, flag, cpt
        
        cptmod10_cond = tf.equal(tf.math.floormod(cpt,10),0)
        err, flag, cpt = tf.cond(cptmod10_cond,cptmod10_true,cptmod10_false)
        
        return flag, err, cpt, Kp, u, v, uprev, vprev
    
    this = tf.while_loop(mycond, loopfn,[flag, err, cpt,  Kp, u, v, uprev, vprev])

    u = this[4]
    v = this[5]
    
    return tf.expand_dims(u,-1)*K*tf.expand_dims(v,-2)

class sinkhorn_knopp_tf_64_class():
    def __init__(self, reg, numItermax=1000, stopThr=1e-9, verbose=False):
        self.err = tf.Variable(1.,dtype=tf.float64,trainable=False,name='err')
        self.cpt = tf.Variable(0,trainable=False,name='cpt')
        self.flag = tf.Variable(1,trainable=False,name='flag')
        self.reg = tf.constant(reg,dtype=tf.float64, name='reg')
        self.numItermax = tf.constant(numItermax, name='numItermax')
        self.stopThr = tf.constant(stopThr,dtype=tf.float64, name='stopThr')
        self.verbose = verbose

    @tf.function
    def __call__(self,a, b, M):
        M = tf.cast(M, tf.float64)
        a = tf.cast(a, tf.float64)
        b = tf.cast(b, tf.float64)

        reg = self.reg
        numItermax = self.numItermax
        stopThr = self.stopThr

        # if reg == 'adaptive':
        #     maxd = tf.reduce_max(M,axis=[-1,-2])
        #     if adaptive_min is None:
        #         reg = maxd/np.float64(708.)
        #     else:
        #         reg = tf.maximum(maxd/np.float64(708.), np.float64(adaptive_min))
        #         #reg = tf.maximum(maxd/np.float64(300.), np.float64(adaptive_min))
        #     K = tf.exp(-M/tf.expand_dims(tf.expand_dims(reg,-1),-1))
        # else:
        K = tf.exp(-M/reg)
        #u = tf.ones(a.shape)/tf.reduce_sum(a,axis=-1,keepdims=True)
        u = tf.fill(tf.shape(a),np.float64(1.))/tf.cast(tf.shape(a)[1],tf.float64)
        v = tf.fill(tf.shape(b),np.float64(1.))/tf.cast(tf.shape(b)[1],tf.float64)
    #     uprev = tf.fill(tf.shape(a),np.float64(1.))/tf.reduce_sum(a,axis=-1,keepdims=True)
    #     vprev = tf.fill(tf.shape(b),np.float64(1.))/tf.reduce_sum(b,axis=-1,keepdims=True)
        uprev = tf.fill(tf.shape(a),np.float64(1.))
        vprev = tf.fill(tf.shape(b),np.float64(1.))
        Kp = tf.expand_dims(1/a,-1)*K

        # err = tf.Variable(1.,dtype=tf.float64,trainable=False)
        # cpt = tf.Variable(0,trainable=False)


        # flag = tf.Variable(1,trainable=False)

        self.flag.assign(1)
        self.cpt.assign(0)
        self.err.assign(1.)

        err = self.err
        cpt = self.cpt
        flag = self.flag
        
        mycond = lambda flag, err, cpt, Kp, u, v, uprev, vprev : tf.logical_and(tf.less(cpt, numItermax),tf.greater(err,stopThr))


        def loopfn(flag, err, cpt, Kp, u, v, uprev, vprev):
            uprev = u
            vprev = v
            
            KtransposeU = tf.squeeze(tf.matmul(K,tf.expand_dims(u,-1),transpose_a=True),axis=-1)
            v = b / (KtransposeU + 1e-300)
            #u = 1/tf.squeeze(tf.matmul(Kp, tf.expand_dims(v,-1)),axis=-1)
            u = a/(tf.squeeze(tf.matmul(K, tf.expand_dims(v,-1)),axis=-1) + 1e-300)
            
    #         error_cond = tf.reduce_any(tf.equal(KtransposeU,0))

            error_cond = tf.reduce_any(tf.math.is_nan(u))
            error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_nan(v)))
            error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_inf(u)))
            error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_inf(v)))
            error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.equal(KtransposeU,0)))
            
            def error_function_true():
                #return tf.Variable(numItermax,trainable=False), uprev, vprev
                tf.print("NAN")
                return numItermax, uprev, vprev
            def error_function_false():
                return cpt+1, u, v
            cpt, u, v = tf.cond(error_cond,error_function_true,error_function_false)

            def cptmod10_true():
                
                tmp2 = tf.squeeze(tf.matmul(tf.expand_dims(u,-2),K),axis=1)*v
                #tmp2 = tf.einsum('ai,aij,aj->aj', u, K, v)
                newerr = tf.norm(tmp2-b,axis=-1)
                stopthr_cond = tf.reduce_all(tf.less(newerr,stopThr))
                
                def stopthr_false():
                    return tf.reduce_mean(newerr), flag + 1, cpt
                def stopthr_true():
                    tf.print("small err", cpt)
                    return tf.reduce_mean(newerr), flag + 1, numItermax
                
                return tf.cond(stopthr_cond,stopthr_true,stopthr_false)
            
            def cptmod10_false():
                return err, flag, cpt
            
            cptmod10_cond = tf.equal(tf.math.floormod(cpt,10),0)
            err, flag, cpt = tf.cond(cptmod10_cond,cptmod10_true,cptmod10_false)
            
            return flag, err, cpt, Kp, u, v, uprev, vprev
        
        this = tf.while_loop(mycond, loopfn,[flag, err, cpt,  Kp, u, v, uprev, vprev])

        u = this[4]
        v = this[5]

        if self.verbose:
            tf.print(this[1])
        
        return tf.expand_dims(u,-1)*K*tf.expand_dims(v,-2)

@tf.function
def sinkhorn_knopp_tf_scaling_64(a, b, M, reg_start, reg_end, numsteps,numItermaxinner, stopThr=1e-9, verbose=False, **kwargs):
    
    numItermax = numItermaxinner * numsteps
    # reg_start = np.float64(reg_start)
    # reg_end = np.float64(reg_end)
    M = tf.cast(M, tf.float64)
    a = tf.cast(a, tf.float64)
    b = tf.cast(b, tf.float64)
    
    
    def getK(M,reg):
        K = tf.exp(-M/reg) + 1e-308
        return K
    
    reg_start_tens = tf.constant(reg_start)
    reg_end_tens = tf.constant(reg_end)
    
    def get_reg(n):
        n = tf.cast(n, tf.float64)
        return (reg_start_tens - reg_end_tens)*(tf.exp(-n) - np.exp(-(numsteps-1))) + reg_end_tens
    
    K = getK(M,reg_start)
    
    reg = tf.Variable(reg_start,dtype=np.float64)
    regstep = tf.constant(np.power(reg_end/reg_start,1./(numsteps-1)), dtype=tf.float64)
    #u = tf.ones(a.shape)/tf.reduce_sum(a,axis=-1,keepdims=True)
    u = tf.fill(tf.shape(a),np.float64(1.))/tf.cast(tf.shape(a)[1],tf.float64)
    v = tf.fill(tf.shape(b),np.float64(1.))/tf.cast(tf.shape(b)[1],tf.float64)
#     uprev = tf.fill(tf.shape(a),np.float64(1.))/tf.reduce_sum(a,axis=-1,keepdims=True)
#     vprev = tf.fill(tf.shape(b),np.float64(1.))/tf.reduce_sum(b,axis=-1,keepdims=True)
    uprev = tf.fill(tf.shape(a),np.float64(1.))
    vprev = tf.fill(tf.shape(b),np.float64(1.))
    #Kp = tf.expand_dims(1/a,-1)*K
    
    err = tf.Variable(1.,dtype=tf.float64, name="err")
    cpt = tf.Variable(0,name="cpt")
    cpt_outer = tf.Variable(0,name='cpt_outer')
    
    flag = tf.Variable(1,name='flag')
    
    mycond = lambda flag, err, cpt, u, v, uprev, vprev, K, reg, cpt_outer : tf.logical_and(tf.less(cpt, numItermax),tf.greater(err,stopThr))


    def loopfn(flag, err, cpt, u, v, uprev, vprev, K, reg, cpt_outer):
        
        cptmod_cond = tf.logical_and(tf.equal(tf.math.floormod(cpt,numItermaxinner),0),
                                    tf.not_equal(cpt,0))
        def cptmod_true():
            newreg = get_reg(cpt_outer)
            return getK(M,newreg), newreg, cpt_outer+1
        
        def cptmod_false():
            return K, reg, cpt_outer
        
        K, reg, cpt_outer = tf.cond(cptmod_cond,cptmod_true, cptmod_false)
        
        
        uprev = u
        vprev = v
        
        KtransposeU = tf.squeeze(tf.matmul(K,tf.expand_dims(u,-1),transpose_a=True),axis=-1)
        v = b / KtransposeU
        #u = 1/tf.squeeze(tf.matmul(Kp, tf.expand_dims(v,-1)),axis=-1)
        u = a/tf.squeeze(tf.matmul(K, tf.expand_dims(v,-1)),axis=-1)
        
#         error_cond = tf.reduce_any(tf.equal(KtransposeU,0))

        error_cond = tf.reduce_any(tf.math.is_nan(u))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_nan(v)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_inf(u)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.is_inf(v)))
        error_cond = tf.logical_or(error_cond, tf.reduce_any(tf.math.equal(KtransposeU,0)))
        
        def error_function_true():
            return tf.Variable(numItermax), uprev, vprev
        def error_function_false():
            return cpt+1, u, v
        cpt, u, v = tf.cond(error_cond,error_function_true,error_function_false)

        
        return flag, err, cpt, u, v, uprev, vprev, K, reg, cpt_outer
    
    this = tf.while_loop(mycond, loopfn,[flag, err, cpt, u, v, uprev, vprev, K, reg, cpt_outer])

    u = this[3]
    v = this[4]
    K = this[-3]
    
    return tf.expand_dims(u,-1)*K*tf.expand_dims(v,-2)


def ground_distance_tf_old(pointsa,pointsb,return_gradients=False,clip=True,epsilon=1e-2):
    
    a_dim = pointsa.shape[-2]
    b_dim = pointsb.shape[-2]
    
    amat = tf.tile(tf.expand_dims(pointsa,2),[1,1,b_dim,1])
    bmat = tf.tile(tf.expand_dims(pointsb,1),[1,a_dim,1,1])
    
    diffmat = bmat - amat
#     return tf.norm(diffmat,axis=3)

    zerogradients = tf.fill(amat.shape,0.)
    if clip:
        clipentries = tf.tile(tf.greater(epsilon, tf.expand_dims(dist,-1)),[1,1,1,2])
        diffmat = tf.where(clipentries,zerogradients,diffmat)
        gradients = tf.where(clipentries, zerogradients, diffmat/tf.expand_dims(dist,-1))
    elif return_gradients:
        gradients = diffmat / tf.expand_dims(dist,-1)

    if return_gradients:
        return tf.norm(diffmat,axis=3), gradients
    else:
        return tf.norm(bmat - amat,axis=3)

def ground_distance_tf(pointsa,pointsb,epsilon=1e-8, mod2pi=True):
    
    # a_dim = pointsa.shape[-2]
    # b_dim = pointsb.shape[-2]

    a_dim = tf.shape(pointsa)[-2]
    b_dim = tf.shape(pointsb)[-2]
    
    amat = tf.tile(tf.expand_dims(pointsa,2),[1,1,b_dim,1])
    bmat = tf.tile(tf.expand_dims(pointsb,1),[1,a_dim,1,1])
    
    diffmat = bmat - amat
    
    if mod2pi:
        deta, dphi = tf.unstack(diffmat,axis=-1)
        dphimod2pi = tf.math.floormod(dphi + math.pi,2*math.pi) - math.pi
        diffmat = tf.stack([dphimod2pi, deta],axis=-1)

    dist = tf.norm(diffmat,axis=3)
    
    
#     return tf.norm(diffmat,axis=3)

    epstensor = tf.constant(epsilon,dtype=tf.float32)

    zerogradients = tf.fill(tf.shape(amat),np.float32(0.))
    
    clipentries = tf.tile(tf.greater(epstensor, tf.expand_dims(dist,-1)),[1,1,1,2])
    diffmat = tf.where(clipentries,zerogradients,diffmat)
    gradients = tf.where(clipentries, zerogradients, diffmat/tf.expand_dims(dist,-1))



    return tf.norm(diffmat,axis=3), gradients

    

def ground_distance_tf_nograd(pointsa,pointsb,epsilon=1e-8, mod2pi=True):
    
    # a_dim = pointsa.shape[-2]
    # b_dim = pointsb.shape[-2]

    a_dim = tf.shape(pointsa)[-2]
    b_dim = tf.shape(pointsb)[-2]
    
    amat = tf.tile(tf.expand_dims(pointsa,2),[1,1,b_dim,1])
    bmat = tf.tile(tf.expand_dims(pointsb,1),[1,a_dim,1,1])
    
    diffmat = bmat - amat
    
    if mod2pi:
        deta, dphi = tf.unstack(diffmat,axis=-1)
        dphimod2pi = tf.math.floormod(dphi + math.pi,2*math.pi) - math.pi
        diffmat = tf.stack([dphimod2pi, deta],axis=-1)


    return tf.norm(diffmat,axis=3)

    
def sinkhorn_loss_tf(in_locations, out_locations, c, out_weights = None, in_weights = None):     
    ground_distance = ground_distance_tf_nograd(in_locations,out_locations)
#self.out_weights = tf.placeholder(tf.int32,shape=([None] + self.n_output)[:-1])
#     if out_weights is None:
#         out_weights = tf.constant(1./500.,shape=out_locations.shape[:-1])
       
    if c.exists_and_is_not_none('adaptive_min'):
        adaptive_min = c.adaptive_min
    else:
        adaptive_min = None
    match = sinkhorn_knopp_tf(in_weights, out_weights, ground_distance, c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr, adaptive_min =adaptive_min)


#     def grad(dL):
#         ground_dist_gradient_perm = tf.transpose(ground_dist_gradient,[0,3,1,2])
#         loss_grad_temp = tf.matrix_diag_part(tf.matmul(tf.tile(tf.expand_dims(match,1),[1,2,1,1]),ground_dist_gradient_perm,transpose_a = True))
#         return tf.transpose(loss_grad_temp,[0,2,1])  

    return tf.linalg.trace(tf.matmul(match,ground_distance,transpose_b=True))

def sinkhorn_loss_tf_scaling(in_locations, out_locations, c, out_weights = None, in_weights = None):     
    ground_distance = ground_distance_tf_nograd(in_locations,out_locations)
#self.out_weights = tf.placeholder(tf.int32,shape=([None] + self.n_output)[:-1])
#     if out_weights is None:
#         out_weights = tf.constant(1./500.,shape=out_locations.shape[:-1])
       
    if c.exists_and_is_not_none('adaptive_min'):
        adaptive_min = c.adaptive_min
    else:
        adaptive_min = None
    match = sinkhorn_knopp_tf_scaling_64(in_weights, out_weights, ground_distance, c.sinkhorn_reg_start, c.sinkhorn_reg_end,numsteps = c.numsteps, numItermaxinner=c.numItermaxinner, stopThr=c.stopThr, adaptive_min =adaptive_min)


#     def grad(dL):
#         ground_dist_gradient_perm = tf.transpose(ground_dist_gradient,[0,3,1,2])
#         loss_grad_temp = tf.matrix_diag_part(tf.matmul(tf.tile(tf.expand_dims(match,1),[1,2,1,1]),ground_dist_gradient_perm,transpose_a = True))
#         return tf.transpose(loss_grad_temp,[0,2,1])  

    return tf.linalg.trace(tf.matmul(tf.cast(match,tf.float32),ground_distance,transpose_b=True))


def greenkhorn_tf(a,b,M,reg=1.,adaptive_min=None,numItermax = 10000,stopThr=None):
    M = tf.cast(M, tf.float64)
    a = tf.cast(a, tf.float64)
    b = tf.cast(b, tf.float64)

    if reg == 'adaptive':
        maxd = tf.reduce_max(M,axis=[-1,-2])
        if adaptive_min is None:
            reg = maxd/np.float64(708.)
        else:
            reg = tf.maximum(maxd/np.float64(708.), np.float64(adaptive_min))
        K = tf.exp(-M/tf.expand_dims(tf.expand_dims(reg,-1),-1))
    else:
        K = tf.exp(-M/reg)
        
    cpt = tf.Variable(0,name='cpt',dtype=tf.int32)

    num_u = tf.shape(a)[1]
    num_v = tf.shape(b)[1]

    u = tf.fill(tf.shape(a),np.float64(1.))/tf.cast(tf.shape(a)[1],tf.float64)
    old_u = tf.fill([tf.shape(a)[0]],np.float64(1.))/tf.cast(tf.shape(a)[1],tf.float64)

    v = tf.fill(tf.shape(b),np.float64(1.))/tf.cast(tf.shape(b)[1],tf.float64)
    old_v = tf.fill([tf.shape(b)[0]],np.float64(1.))/tf.cast(tf.shape(b)[1],tf.float64)
    G = tf.expand_dims(u,-1)*K*tf.expand_dims(v,-2)

    viol_1 = tf.reduce_sum(G,axis = 2) - a
    viol_2 = tf.reduce_sum(G,axis = 1) - b
    stopThr_val = tf.fill(tf.shape(a)[0:1],np.float64(1.))



    def loop_fn(cpt,u,v,G,viol_1,viol_2,old_u,old_v):
        i_1 = tf.argmax(tf.abs(viol_1),axis=-1)
        i_2 = tf.argmax(tf.abs(viol_2),axis=-1)
        #m_viol_1 = tf.reduce_max(tf.abs(viol),axis=-1)
        row_indices_1 = tf.cast(tf.range(tf.shape(i_1)[0]),tf.int64)
        full_indices_1 = tf.stack([row_indices_1, i_1], axis=1)
        m_viol_1 = tf.abs(tf.gather_nd(viol_1,full_indices_1))
        #m_viol_2 = tf.reduce_max(tf.abs(viol_2),axis=-1)
        row_indices_2 = tf.cast(tf.range(tf.shape(i_2)[0]),tf.int64)
        full_indices_2 = tf.stack([row_indices_2, i_2], axis=1)
        m_viol_2 = tf.abs(tf.gather_nd(viol_2,full_indices_2))
        stopThr_val = tf.maximum(m_viol_1,m_viol_2)

        u_index_cond = tf.equal(tf.tile(tf.expand_dims(tf.cast(tf.range(tf.shape(viol_1)[1]),tf.int64),0),[tf.shape(viol_1)[0],1]),tf.expand_dims(i_1,-1))
        m1_gtr_m2_cond = tf.greater(tf.expand_dims(m_viol_1,-1),tf.expand_dims(m_viol_2,-1))
        cond_u = tf.logical_and(m1_gtr_m2_cond,
                                u_index_cond)


        old_u = tf.gather_nd(u,full_indices_1)
        agather = tf.gather_nd(a,full_indices_1)
        Kgather = tf.gather_nd(K,full_indices_1)

        new_u = tf.expand_dims(agather/tf.squeeze(tf.matmul(tf.expand_dims(Kgather,-2),tf.expand_dims(v,-1)),axis=[-1,-2]),-1)
        u = tf.where(cond_u,tf.tile(new_u,[1,num_u]),u)

        ugather = tf.gather_nd(u,full_indices_1)

        new_G = tf.expand_dims(tf.expand_dims(ugather,-1)*Kgather*v,1)
        G = tf.where(tf.tile(tf.expand_dims(cond_u,-1),[1,1,num_v]),tf.tile(new_G,[1,num_u,1]),G)

        new_viol = ugather*tf.squeeze(tf.matmul(tf.expand_dims(Kgather,-2),tf.expand_dims(v,-1)),axis=[-1,-2]) - agather
        viol_1 = tf.where(cond_u,tf.tile(tf.expand_dims(new_viol,-1),[1,num_u]),viol_1)

        viol_2 = tf.where(tf.tile(m1_gtr_m2_cond,[1,num_v]),viol_2 + Kgather * tf.expand_dims(ugather - old_u,-1) * v,viol_2)


        v_index_cond = tf.equal(tf.tile(tf.expand_dims(tf.cast(tf.range(tf.shape(viol_2)[1]),tf.int64),0),[tf.shape(viol_2)[0],1]),tf.expand_dims(i_2,-1))
        cond_v = tf.logical_and(tf.logical_not(m1_gtr_m2_cond),
                                v_index_cond)

        old_v = tf.gather_nd(v,full_indices_2)
        bgather = tf.gather_nd(b,full_indices_2)
        Kgather_2 = tf.gather_nd(tf.transpose(K,[0,2,1]),full_indices_2)

        new_v = tf.expand_dims(bgather/tf.squeeze(tf.matmul(tf.expand_dims(Kgather_2,-2),tf.expand_dims(u,-1)),axis=[-1,-2]),-1)
        v = tf.where(cond_v,tf.tile(new_v,[1,num_v]),v)

        vgather = tf.gather_nd(v,full_indices_2)


        new_G = tf.expand_dims(tf.expand_dims(vgather,-1)*Kgather_2*u,2)
        G = tf.where(tf.tile(tf.expand_dims(cond_v,-2),[1,num_u,1]),tf.tile(new_G,[1,1,num_v]),G)

        new_viol = vgather*tf.squeeze(tf.matmul(tf.expand_dims(Kgather_2,-2),tf.expand_dims(u,-1)),axis=[-1,-2]) - bgather
        viol_2 = tf.where(cond_v,tf.tile(tf.expand_dims(new_viol,-1),[1,num_v]),viol_2)

        viol_1 = tf.where(tf.tile(m1_gtr_m2_cond,[1,num_u]),viol_1 + Kgather_2 * tf.expand_dims(vgather - old_v,-1) * u,viol_1)

        cpt = cpt+1

        return cpt,u,v,G,viol_1,viol_2,old_u,old_v

    def cond(cpt,u,v,G,viol_1,viol_2,old_u,old_v):
        return tf.less(cpt,numItermax)

    this= tf.while_loop(cond,loop_fn,[cpt,u,v,G,viol_1,viol_2,old_u,old_v])
    
    return this[3]



def greenkhorn_loss_tf(in_locations, out_locations, c, out_weights = None, in_weights = None):     
    ground_distance = ground_distance_tf_nograd(in_locations,out_locations)
#self.out_weights = tf.placeholder(tf.int32,shape=([None] + self.n_output)[:-1])
    if out_weights is None:
        out_weights = tf.constant(1./500.,shape=out_locations.shape[:-1])
       
    if c.exists_and_is_not_none('adaptive_min'):
        adaptive_min = c.adaptive_min
    else:
        adaptive_min = None
    match = greenkhorn_tf(in_weights, out_weights, ground_distance, c.sinkhorn_reg, numItermax=c.numItermax, stopThr=c.stopThr, adaptive_min =adaptive_min)


#     def grad(dL):
#         ground_dist_gradient_perm = tf.transpose(ground_dist_gradient,[0,3,1,2])
#         loss_grad_temp = tf.matrix_diag_part(tf.matmul(tf.tile(tf.expand_dims(match,1),[1,2,1,1]),ground_dist_gradient_perm,transpose_a = True))
#         return tf.transpose(loss_grad_temp,[0,2,1])  

    return tf.linalg.trace(tf.matmul(tf.cast(match,tf.float32),ground_distance,transpose_b=True))

    M = tf.cast(M, tf.float64)
    a = tf.cast(a, tf.float64)
    b = tf.cast(b, tf.float64)
    if reg == 'adaptive':
        maxd = tf.reduce_max(M,axis=[-1,-2])
        if adaptive_min is None:
            reg = maxd/np.float64(708.)
        else:
            reg = tf.maximum(maxd/np.float64(708.), np.float64(adaptive_min))
            #reg = tf.maximum(maxd/np.float64(300.), np.float64(adaptive_min))
        K = tf.exp(-M/tf.expand_dims(tf.expand_dims(reg,-1),-1))
    else:
        K = tf.exp(-M/reg)



class sinkhorn_knopp_tf_stabilized_class():
    def __init__(self, reg,
    tau_val = 1e3,
    numItermax=100,
    stopThr=1e-5,
    check_err_period = 5,
    dtype=tf.float32,
    ret_alpha_beta = False,
    verbose=False):
        # with tf.name_scope("sinkhorn_knopp_tf_stabilized") as scope:
        self.dtype = dtype
        self.reg = tf.Variable(reg,dtype=dtype,trainable=False,name='reg')
        self.err = tf.Variable(1.,dtype=dtype,trainable=False,name='err')
        self.cpt = tf.Variable(0,trainable=False,name='cpt')
        self.loop = tf.Variable(True,trainable=False,name='loop')
        self.tau = tf.constant(tau_val,name = 'tau',dtype=dtype)
        self.numItermax = tf.constant(numItermax,name='numitermax')
        self.stopThr = tf.constant(stopThr, name="stopThr",dtype=dtype)
        self.check_err_period = tf.constant(check_err_period,name="check_err_period")
        self.ret_alpha_beta = ret_alpha_beta
        self.verbose = verbose
        if self.dtype is tf.float64:
            self.EPSILON = 1e-100
        else:
            self.EPSILON = 1e-30

    @tf.function
    def __call__(self,a, b, M, alpha, beta):

        M = tf.cast(M,self.dtype)
        b = tf.cast(b,self.dtype)
        a = tf.cast(a,self.dtype)

        # init data
        shape_a = tf.shape(a)
        shape_b = tf.shape(b)
        dim_a = shape_a[1]
        dim_b = shape_b[1]



        # # we assume that no distances are null except those of the diagonal of
        # # distances
        # if warmstart is None:
        #     alpha = tf.fill(tf.shape(a),np.float32(0.))
        #     beta = tf.fill(tf.shape(b),np.float32(0.))
        # else:
        # alpha, beta = warmstart
        #err = tf.cast(tf.fill([dim_a],1.),self.dtype)

        u_init = tf.cast(tf.fill(shape_a,1.),self.dtype)/tf.cast(dim_a,self.dtype)
        v_init = tf.cast(tf.fill(shape_b,1.),self.dtype)/tf.cast(dim_b,self.dtype)

        u = u_init
        v = v_init

        @tf.function
        def get_K(alpha, beta):
            """log space computation"""
            return tf.exp(-(M - tf.expand_dims(alpha,-1) -  tf.expand_dims(beta,-2)) / self.reg) + 1e-308

        @tf.function
        def get_Gamma(alpha, beta, u, v):
            """log space gamma computation"""
            return tf.exp(-(M - tf.expand_dims(alpha,-1) -  tf.expand_dims(beta,-2)) / self.reg + tf.math.log(tf.expand_dims(u,-1)) + tf.math.log(tf.expand_dims(v,-2)))

        # print(np.min(K))

        K = get_K(alpha, beta)
        transp = K
        self.loop.assign(True)
        # self.cpt.assign(0)
        # cpt = tf.Variable(0,trainable=False)
        

        uprev = u_init
        vprev = v_init

        for cpt in tf.range(self.numItermax):

            # self.cpt.assign(cpt)

            uprev = u
            vprev = v


            KtransposeU = tf.squeeze(tf.matmul(K,tf.expand_dims(u,-1),transpose_a=True),axis=-1)

            #sinkhorn update
            v = b /(KtransposeU + self.EPSILON)
            u = a/(tf.squeeze(tf.matmul(K, tf.expand_dims(v,-1)),axis=-1) + self.EPSILON)


            machine_err = tf.math.reduce_any(tf.math.is_nan(u))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_nan(v)))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_inf(v)))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_inf(u)))

            if machine_err:
                # we have reached the machine precision
                # come back to previous solution and quit loop
                tf.print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break

            # remove numerical problems and store them in K
            if tf.logical_or(tf.greater(tf.reduce_max(tf.math.abs(u)),self.tau), tf.greater(tf.reduce_max(tf.math.abs(v)),self.tau)):
                alpha = alpha + self.reg * tf.math.log(u)
                beta = beta + self.reg * tf.math.log(v)
                u = u_init
                v = v_init
                K = get_K(alpha, beta)
                

            if tf.logical_and(tf.greater(cpt,0),tf.equal(tf.math.floormod(cpt,self.check_err_period),0)):
                # we can speed up the process by checking for the error only all
                # the 10th iterations

                transp = get_Gamma(alpha, beta, u, v)
                self.err.assign(tf.reduce_mean(tf.norm((tf.reduce_sum(transp, axis=1) - b),axis=-1)))
                #avg_err = tf.reduce_mean(tf.norm((tf.reduce_sum(transp, axis=1) - b),axis=-1))

                if tf.less_equal(self.err, self.stopThr):
                    if self.verbose:
                        tf.print("small error", cpt)
                    self.loop.assign(False)
                    break


        if self.verbose:
            tf.print(self.err)     
        
        if self.ret_alpha_beta:
            return get_Gamma(alpha, beta, u, v), alpha + self.reg * tf.math.log(u), beta + self.reg*tf.math.log(v)
        else:
            return get_Gamma(alpha, beta, u, v)


class sinkhorn_knopp_tf_stabilized_alt_class():
    def __init__(self, reg,
    tau_val = 1e3,
    numItermax=100,
    stopThr=1e-5,
    check_err_period = 10,
    dtype=tf.float32,
    ret_alpha_beta = False,
    verbose=False,
    warmstart = True):
        # with tf.name_scope("sinkhorn_knopp_tf_stabilized") as scope:
        self.dtype = dtype
        self.reg = tf.Variable(reg,dtype=dtype,trainable=False,name='reg')
        #self.reg = tf.constant(reg,dtype=tf.float64, name='reg')
        self.err = tf.Variable(1.,dtype=dtype,trainable=False,name='err')
        self.cpt = tf.Variable(0,trainable=False,name='cpt')
        self.loop = tf.Variable(True,trainable=False,name='loop')
        self.tau = tf.constant(tau_val,name = 'tau',dtype=dtype)
        self.numItermax = tf.constant(numItermax,name='numitermax')
        self.stopThr = tf.constant(stopThr, name="stopThr",dtype=dtype)
        self.check_err_period = tf.constant(check_err_period,name="check_err_period")
        self.ret_alpha_beta = ret_alpha_beta
        self.verbose = verbose
        if self.dtype is tf.float64:
            self.EPSILON = 1e-100
        else:
            self.EPSILON = 1e-30
        self.warmstart = warmstart

    # @tf.function
    # def __call__(self,a, b, G):
    #     return self.do_dense(a, b, G)

    def __call__(self,a, b, G,sparse = False):
        if sparse:
            return self.do_sparse(a, b, G)
        else:
            return self.do_dense(a, b, G)

    @tf.function
    def do_dense(self,a, b, G):

        # M = tf.cast(M,self.dtype)
        # b = tf.cast(b,self.dtype)
        # a = tf.cast(a,self.dtype)

        # init data
        shape_a = tf.shape(a)
        shape_b = tf.shape(b)
        dim_a = shape_a[1]
        dim_b = shape_b[1]

        u_init = tf.cast(tf.fill(shape_a,1.),self.dtype)/tf.cast(dim_a,self.dtype)
        v_init = tf.cast(tf.fill(shape_b,1.),self.dtype)/tf.cast(dim_b,self.dtype)

        u_repl = tf.cast(tf.fill(shape_a,1.),self.dtype)
        v_repl = tf.cast(tf.fill(shape_b,1.),self.dtype)

        u = u_init
        v = v_init

        reg = self.reg
        numItermax = self.numItermax
        stopThr = self.stopThr

        # print(np.min(K))
        # if self.warmstart:
        #     K = get_K(alpha, beta)
        # else:
        #     K = tf.exp(-M/reg)
        # transp = K
        K = G
        self.loop.assign(True)
        self.cpt.assign(0)
        
        err = self.err
        cpt = self.cpt
        loop = self.loop
        
        uprev = u_init
        vprev = v_init

        def loopfn(loop, err, cpt, u, v, uprev, vprev):

            uprev = u
            vprev = v


            KtransposeU = tf.squeeze(tf.matmul(K,tf.expand_dims(u,-1),transpose_a=True),axis=-1)

            #sinkhorn update
            v = b /(KtransposeU + self.EPSILON)
            u = a/(tf.squeeze(tf.matmul(K, tf.expand_dims(v,-1)),axis=-1) + self.EPSILON)


            machine_err = tf.math.reduce_any(tf.math.is_nan(u))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_nan(v)))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_inf(v)))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_inf(u)))

            def machine_error_true():
                return tf.constant(False)
            def machine_error_false():
                return tf.constant(True)

            loop = tf.cond(machine_err,machine_error_true,machine_error_false)

           
            check_err_cond = tf.equal(tf.math.floormod(cpt,self.check_err_period),0)

            def check_err_true():
            
                transp = tf.expand_dims(u,-1)*K*tf.expand_dims(v,-2)
                newerr = tf.reduce_mean(tf.norm((tf.reduce_sum(transp, axis=1) - b),axis=-1))
                stopthr_cond = tf.less(newerr,self.stopThr)

                
                def stopthr_false():
                    return newerr, tf.constant(True)
                def stopthr_true():
                    return newerr, tf.constant(False)
                
                return tf.cond(stopthr_cond,stopthr_true,stopthr_false)

            def check_err_false():
                return err, tf.constant(True)

            err, loop = tf.cond(check_err_cond,check_err_true, check_err_false)
        
            return loop, err, cpt, u, v, uprev, vprev

        loop_cond = lambda loop, err, cpt, u, v, uprev, vprev : loop

        loop, err, cpt, u, v, uprev, vprev = tf.while_loop(loop_cond,loopfn,[self.loop, self.err, self.cpt, u, v, uprev, vprev],maximum_iterations=self.numItermax)


        if self.verbose:
            tf.print(err)     
        
        return tf.expand_dims(u,-1)*K*tf.expand_dims(v,-2)

    @tf.function
    def do_sparse(self,a, b, G):
        # M = tf.cast(M,self.dtype)
        # b = tf.cast(b,self.dtype)
        # a = tf.cast(a,self.dtype)

        # init data
        shape_a = tf.shape(a)
        shape_b = tf.shape(b)
        dim_a = shape_a[1]
        dim_b = shape_b[1]

        u_init = tf.cast(tf.fill(shape_a,1.),self.dtype)/tf.cast(dim_a,self.dtype)
        v_init = tf.cast(tf.fill(shape_b,1.),self.dtype)/tf.cast(dim_b,self.dtype)

        u_repl = tf.cast(tf.fill(shape_a,1.),self.dtype)
        v_repl = tf.cast(tf.fill(shape_b,1.),self.dtype)

        u = u_init
        v = v_init

        reg = self.reg
        numItermax = self.numItermax
        stopThr = self.stopThr

        # print(np.min(K))
        # if self.warmstart:
        #     K = get_K(alpha, beta)
        # else:
        #     K = tf.exp(-M/reg)
        # transp = K
        K = G
        self.loop.assign(True)
        self.cpt.assign(0)
        
        err = self.err
        cpt = self.cpt
        loop = self.loop
        
        uprev = u_init
        vprev = v_init

        def loopfn(loop, err, cpt, u, v, uprev, vprev):

            uprev = u
            vprev = v


            KtransposeU = tf.sparse.reduce_sum(K * u[:,:,None],axis=-2)

            #sinkhorn update
            v = b /(KtransposeU + self.EPSILON)
            
            u = a/(tf.sparse.reduce_sum(K * v[:,None,:],axis=-1) + self.EPSILON)

            u = tf.reshape(u, tf.shape(a))
            v = tf.reshape(v, tf.shape(b))


            machine_err = tf.math.reduce_any(tf.math.is_nan(u))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_nan(v)))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_inf(v)))
            machine_err = tf.logical_or(machine_err,tf.math.reduce_any(tf.math.is_inf(u)))

            def machine_error_true():
                return tf.constant(False)
            def machine_error_false():
                return tf.constant(True)

            loop = tf.cond(machine_err,machine_error_true,machine_error_false)

           
            check_err_cond = tf.equal(tf.math.floormod(cpt,self.check_err_period),0)

            def check_err_true():

                transp = u[:,:,None]*K*v[:,None,:]
                newerr = tf.reduce_mean(tf.norm((tf.sparse.reduce_sum(transp, axis=1) - b),axis=-1))
                stopthr_cond = tf.less(newerr,self.stopThr)

                
                def stopthr_false():
                    return newerr, tf.constant(True)
                def stopthr_true():
                    return newerr, tf.constant(False)
                
                return tf.cond(stopthr_cond,stopthr_true,stopthr_false)

            def check_err_false():
                return err, tf.constant(True)

            err, loop = tf.cond(check_err_cond,check_err_true, check_err_false)
        
            

            return [loop, err, cpt, u, v, uprev, vprev]

        loop_cond = lambda loop, err, cpt, u, v, uprev, vprev : loop

        loop, err, cpt, u, v, uprev, vprev = tf.while_loop(loop_cond,loopfn,[self.loop, self.err, self.cpt, u, v, uprev, vprev],maximum_iterations=self.numItermax)


        if self.verbose:
            tf.print(err)

        transp = u[:,:,None]*K*v[:,None,:]

        return transp


class sinkhorn_knopp_tf_scaling_stabilized_class():
    def __init__(self,
                reg_init,
                reg_final,
                tau_val = 1e3,
                numItermaxinner=100,
                numIter = 10,
                stopThr=1e-5,
                check_err_period = 10,
                dtype=tf.float32,
                verbose=False,
                sparse = False,
                sparse_min = 1e-10,
                numdense = 4):
        with tf.name_scope("sinkhorn_knopp_tf_scaling_stabilized") as scope:
            self.dtype = dtype
            self.reg_init = tf.Variable(reg_init,dtype=dtype,trainable=False,name='reg_init')
            self.reg = tf.Variable(reg_init,dtype=dtype,trainable=False,name='reg')
            self.reg_final = tf.Variable(reg_final,dtype=dtype,trainable=False,name='reg_final')
            self.err = tf.Variable(1.,dtype=dtype,trainable=False,name='err')
            self.cpt = tf.Variable(0,trainable=False,name='cpt')
            self.loop = tf.Variable(True,trainable=False,name='loop')
            self.tau = tf.constant(tau_val,name = 'tau',dtype=dtype)
            self.numItermaxinner = tf.constant(numItermaxinner,name='numitermax')
            self.numIter = tf.constant(numIter,name='numitermax')
            self.stopThr = tf.constant(stopThr, name="stopThr",dtype=dtype)
            self.check_err_period = tf.constant(check_err_period,name="check_err_period")
            self.reg_step = tf.math.pow(self.reg_final/self.reg_init,1./tf.cast(numIter-1,self.dtype)) 
            self.sink_fn = sinkhorn_knopp_tf_stabilized_alt_class(self.reg,numItermax=numItermaxinner,stopThr=stopThr,check_err_period=check_err_period,dtype=dtype,ret_alpha_beta = True,verbose=verbose)
            self.verbose = verbose
            self.sparse = sparse
            self.sparse_min = sparse_min
            self.numdense = numdense

    def __call__(self,a, b, M):
        if self.sparse is True:
            return self.do_sparse(a, b, M)
        else:
            return self.do_dense(a, b, M)

    @tf.function
    def do_dense(self,a,b,M):
        # init data
        shape_a = tf.shape(a)
        shape_b = tf.shape(b)
        dim_a = shape_a[1]
        dim_b = shape_b[1]

        a = tf.cast(a,self.dtype)
        b = tf.cast(b,self.dtype)
        M = tf.cast(M,self.dtype)


        self.reg.assign(self.reg_init)
        G = tf.exp(-M/self.reg)
        next_reg = self.reg

        self.sink_fn.reg.assign(self.reg)
        G = self.sink_fn(a, b, G, sparse=False)

       
        for cpt in tf.range(1,self.numIter):

            next_reg = self.reg * self.reg_step
            G = tf.pow(G,self.reg/next_reg)
            self.reg.assign(next_reg)

            self.sink_fn.reg.assign(self.reg)

            G = self.sink_fn(a, b, G, sparse = False)



        return G

            

    
    @tf.function
    def do_sparse(self,a,b,M):
        # init data
        shape_a = tf.shape(a)
        shape_b = tf.shape(b)
        dim_a = shape_a[1]
        dim_b = shape_b[1]

        # self.cpt.assign(0)

        a = tf.cast(a,self.dtype)
        b = tf.cast(b,self.dtype)
        M = tf.cast(M,self.dtype)


        self.reg.assign(self.reg_init)
        G = tf.exp(-M/self.reg)
        next_reg = self.reg

        self.sink_fn.reg.assign(self.reg)
        G = self.sink_fn(a, b, G, sparse=False)

        for cpt in tf.range(1,self.numdense):

            next_reg = self.reg * self.reg_step
            G = tf.pow(G,self.reg/next_reg)
            self.reg.assign(next_reg)

            self.sink_fn.reg.assign(self.reg)

            G = self.sink_fn(a, b, G, sparse=False)

        if self.sparse:
            G = tf.where(tf.less(G, self.sparse_min), tf.zeros_like(G), G)
            G = tf.sparse.from_dense(G)


        
        for cpt in tf.range(self.numdense,self.numIter):

            next_reg = self.reg * self.reg_step
            if self.sparse:
                G = tf.SparseTensor(G.indices,tf.pow(G.values,self.reg/next_reg),G.dense_shape)
            else:
                G = tf.pow(G,self.reg/next_reg)
            self.reg.assign(next_reg)

            self.sink_fn.reg.assign(self.reg)

            G = self.sink_fn(a, b, G, sparse = self.sparse)



        return G

            

