import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.linalg import sqrtm
from numpy import trace
from keras import backend as K

def mixup(X_train, y_train, bs = 10, alpha=0.1,):
    #print(bs)
    x1 = X_train[:bs]
    x2 = X_train[bs:]
    # print(x1.shape)

    y1 = y_train[:bs]
    y2 = y_train[bs:]

    lam = np.random.beta(alpha, alpha, 1)
    # print(x1.shape)
    x = (lam * x1 + (1. - lam) * x2)
    y = (lam * y1 + (1. - lam) * y2)

    return x, y


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                #print('reinitializing layer {}.{}'.format(layer.name, v))

def batch_generator(data, batch_size, replace = False):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=replace)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


def gau_js_samples(sample0, sample1):
    pm = np.mean(sample0, axis=0)
    #print(pm.shape)
    #exit()
    pv = np.cov(sample0.T)
    pv = pv + np.random.random(size = pv.shape)*0.2
    qm = np.mean(sample1, axis=0)
    qv = np.cov(sample1.T)
    qv = qv + np.random.random(size=qv.shape) * 0.2

    #print(pm, pv, qm, qv)
    #print("==============")
    #print(pm.max(), qm.max())
    #exit()
    print(np.linalg.norm(pv*pm - qv*qm))
    return gau_js(pm, pv, qm, qv)



def bh_distance(sample0, sample1):
    pm = np.mean(sample0, axis=0)
    #print(pm.shape)
    #exit()
    pv = np.cov(sample0.T)
    pv = pv + np.random.random(size = pv.shape)*0.2
    qm = np.mean(sample1, axis=0)
    qv = np.cov(sample1.T)
    qv = qv + np.random.random(size=qv.shape) * 0.2

    p = (pv + qv) / 2
    diff = np.array([(qm - pm)])
    print(np.linalg.det(p))

    distance = (1/8.0) *diff.dot(np.linalg.inv(p)).dot(diff.T) + 0.5 * np.log(np.linalg.det(p)/(np.linalg.det(pv) * np.linalg.det(qv)))
    print(distance, np.linalg.det(pv), np.linalg.det(qv))
    return distance


def wasser(sample0, sample1):
    pm = np.mean(sample0, axis=0)
    pv = np.cov(sample0.T)
    qm = np.mean(sample1, axis=0)
    qv = np.cov(sample1.T)

    #distance = np.linalg.norm(pm - qm) + np.linalg.norm(sqrtm(pv) - sqrtm(qv))
    s1 = sqrtm(pv).dot(qv).dot(sqrtm(pv))
    #print(s1)
    #exit()
    s2 = (pv + qv) - 2 * sqrtm(s1)
    #print(s2)

    distance = np.linalg.norm(pm - qm) + trace(s2)
    # print(distance)
    # print("====")

    return distance



def geu_kl(pm, pv, qm, qv):


    dpv = np.linalg.det(pv)
    dqv = np.linalg.det(qv)
    #print(dpv)
    # Inverses of diagonal covariances pv, qv
    iqv = np.linalg.inv(qv)
    #ipv = np.linalg.inv(pv)
    # Difference between means pm, qm
    diff = qm - pm
    # KL(p||q)
    kl = (0.5 *
           (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
            + (np.matrix.trace(iqv.T.dot(pv)))         # + tr(\Sigma_q^{-1} * \Sigma_p)
            + diff.dot(iqv).dot(diff) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)))

    # print(dqv, dqv, pv.shape, qv.shape)
    # print(np.log(dqv / dpv))
    # print((np.matrix.trace(iqv.T.dot(pv))))
    # print(diff.dot(iqv).dot(diff))
    #
    # print(len(pm))
    # print(kl)
    # print("=========")

    return kl

def gau_js(pm, pv, qm, qv):
    """
    Jensen-Shannon divergence between two Gaussians.  Also computes JS
    divergence between a single Gaussian pm,pv and a set of Gaussians
    qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """

    return 0.5 * (geu_kl(pm, pv, qm, qv) + geu_kl(qm, qv, pm, pv))



def dump_csv(filename, data, columns):
    #print(len(data))
    df = pd.DataFrame(columns = columns, data=data)
    df.to_csv(filename)


def entropy1(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


from keras import backend as K
from keras.legacy import interfaces
from keras.optimizers import Adam
from keras.callbacks import Callback


class DecoupleWeightDecay:
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        weight_decay: weight decay value that will be mutltiplied to the parameter
    # References
        - [AdamW - DECOUPLED WEIGHT DECAY REGULARIZATION](
           https://arxiv.org/pdf/1711.05101.pdf)

    """

    def __init__(self, weight_decay, **kwargs):
        with K.name_scope(self.__class__.__name__):
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
        super(DecoupleWeightDecay, self).__init__(**kwargs)

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        updates = super(DecoupleWeightDecay, self).get_updates(loss, params)
        #TODO change loop to vectorized
        for p in params:
            updates.append(K.update_sub(p, self.weight_decay*p))
        return updates


def create_decouple_optimizer(optimizer):
    class OptimizerW(DecoupleWeightDecay, optimizer):
        def __init__(self, weight_decay, **kwargs):
            super(OptimizerW, self).__init__(weight_decay, **kwargs)


class WeightDecayScheduler(Callback):
    def __init__(self, init_lr):
        super(WeightDecayScheduler, self).__init__()
        self.previous_lr = init_lr


    def on_epoch_begin(self, epoch, logs=None):
        current_lr = float(K.get_value(self.model.optimizer.lr))
        coeff = current_lr / self.previous_lr
        new_weight_decay = float(K.get_value(self.model.optimizer.weight_decay)) * coeff
        K.set_value(self.model.optimizer.weight_decay, new_weight_decay)
        self.previous_lr = current_lr
        if coeff!=1:
            print(epoch, coeff)

    def on_epoch_end(self, epoch, logs=None):
        return


class AdamW(DecoupleWeightDecay, Adam):
    def __init__(self, weight_decay, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(AdamW, self).__init__(weight_decay=weight_decay, lr=lr, beta_1=beta_1, beta_2=beta_2,
                 epsilon=epsilon, decay=decay, amsgrad=amsgrad, **kwargs)


from keras.optimizers import Optimizer
from keras import backend as K
import six
import copy
from six.moves import zip
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces


class SGDW(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        weight_decay: float >= 0. Decoupled weight decay over each update.
    # References
        - [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html)
        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., weight_decay=1e-4,  # decoupled weight decay (1/6)
                 nesterov=False, **kwargs):
        super(SGDW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.init_lr = lr  # decoupled weight decay (2/6)
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.wd = K.variable(weight_decay, name='weight_decay')  # decoupled weight decay (3/6)
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd  # decoupled weight decay (4/6)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        eta_t = lr / self.init_lr  # decoupled weight decay (5/6)

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g - eta_t * wd * p  # decoupled weight decay (6/6)
            else:
                new_p = p + v - lr * wd * p  # decoupled weight decay

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGDW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer

import tensorflow as tf


def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma


def log_mixture_prior_prob(w):
    comp_1_dist = tf.distributions.Normal(0.0, prior_params[0])
    comp_2_dist = tf.distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))


# Mixture prior parameters shared across DenseVariational layer instances
prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)


class DenseVariational(Layer):
    def __init__(self, output_dim, kl_loss_weight, activation=None, **kwargs):
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._trainable_weights.append(prior_params)

        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.output_dim),
                                         initializer=initializers.normal(stddev=prior_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.output_dim,),
                                       initializer=initializers.normal(stddev=prior_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.output_dim),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.output_dim,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.nn.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random_normal(self.kernel_mu.shape)

        bias_sigma = tf.nn.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random_normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tf.distributions.Normal(mu, sigma)
        return self.kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w))