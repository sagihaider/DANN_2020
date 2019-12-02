import tensorflow.keras.backend as K
import numpy as np
from keras.constraints import MaxNorm
from keras.layers import GaussianNoise
from keras.layers import Input, Activation, Dropout
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
#from tensorflow.
from keras.activations import selu
from keras.layers.noise import AlphaDropout
from keras.layers.core import ActivityRegularization
from keras.losses import categorical_hinge, categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2,l1
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from tensorflow import Graph, Session

from utils import batch_generator
from utils import dump_csv
from utils import AdamW, SGDW, mixup, DenseVariational
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.layers import BatchNormalization


class NNClassifier(BaseEstimator, ClassifierMixin):
    """A neural network classifier that can be used for all things """

    def __init__(self, n_iterations, Xt, enable_dann, do_strength=10, batch_size=128, plot=False, mixup=False):
        self.n_iterations = n_iterations
        self.Xt = Xt
        self.graph = Graph()
        self.enable_dann = enable_dann
        with self.graph.as_default():
            self.session = Session()
            with self.session.as_default():
                self.model, self.source_classification_model, self.domain_classification_model, self.embeddings_model, self.domain_classification_model_stand_alone = build_models(
                    do_strength)

        self.do_strength = do_strength
        self.batch_size = batch_size
        self.plot = plot
        self.mixup = mixup

        self.models = self.model, self.source_classification_model, self.domain_classification_model, self.embeddings_model, self.domain_classification_model_stand_alone

    def get_params(self, deep=True):

        return {"n_iterations": self.n_iterations,
                "Xt": self.Xt,
                "enable_dann": self.enable_dann,
                "do_strength": self.do_strength,
                "batch_size": self.batch_size,
                "plot": self.plot,
                "mixup":self.mixup
                }

    def fit(self, X, y=None, sample_weight=None, features = None):

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.features = features

        with self.graph.as_default():
            with self.session.as_default():
                if sample_weight is not None:
                    X_train, y_train = self.resample_with_replacement(X, y, sample_weight)
                    self.darken = train(X_train, y_train, self.Xt, self.n_iterations, self.models,
                                        self.enable_dann, self.batch_size, self.plot, self.mixup, self.features)
                else:
                    self.darken = train(X, y, self.Xt, self.n_iterations, self.models, self.enable_dann,
                                        self.batch_size, self.plot, self.mixup, self.features)

        return self

    def predict(self, X, y=None):
        with self.graph.as_default():
            with self.session.as_default():
                return self.source_classification_model.predict(X, batch_size=1000).argmax(1)

    def predict_proba(self, X, y=None):
        with self.graph.as_default():
            with self.session.as_default():
                return self.source_classification_model.predict(X, batch_size=1000)

    def predict_embs(self, X, y=None):
        with self.graph.as_default():
            with self.session.as_default():
                return self.embeddings_model.predict(X, batch_size=1000)

    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled


def entropy(y):
    # Clip to avoid Zeros.
    y = K.clip(y, 1e-20, 1.0)
    return -K.sum(y * K.log(y))


def custom_loss(y_true, y_pred, beta=0.1):
    return categorical_crossentropy(y_true, y_pred)  # + beta * entropy(y_pred)
    #return categorical_hinge(y_true, y_pred)
#
# def get_adam_w():
# #     #return SGD(lr=0.01)  #
# #     #return SGDW(lr = 0.01, weight_decay=0.01)
#       return Adam(lr=0.00041)
# #     #return AdamW(lr=0.001, weight_decay=0.01, clipnorm=0.1 )
# #     # from tensorflow.contrib.opt import AdamWOptimizer
# #     # fopt = AdamWOptimizer(weight_decay=0.001, learning_rate=.001)
# #     # #fopt = ShampooWOptimizer(weight_decay=0.001, learning_rate=0.001)
# #     # return fopt
# #     #return SGD(lr=0.01)

def get_optimizer():
    return SGD(lr=0.001)  #
    #return SGDW(lr = 0.01, weight_decay=0.01)
    #return Adam(lr=0.001)
    #return AdamW(lr=0.0001, weight_decay=0.01, clipnorm=0.1 )
    #from tensorflow.contrib.opt import AdamWOptimizer
    #fopt = AdamWOptimizer(weight_decay=0.00001, learning_rate=.00001)
    #fopt = ShampooWOptimizer(weight_decay=0.001, learning_rate=0.001)
    #return fopt
    #from tensorflow.trainining import

# #
def rl():
    return l2(0.01)#
    #return None  #


def cn():
    return None
    #return MaxNorm(0.5)


def build_models(do_strength):
    """Creates three different models, one used for source only training, two used for domain adaptation"""
    EMB_inputs = Input(shape=(20,))  # We have 20 variables in the EEG data from FBCSP
    EMB = EMB_inputs
    #EMB = GaussianNoise(0.5)(EMB)
    #EMB = Dropout(0.1)(EMB)

    a = get_optimizer()

    n_neurons = 2 #

    # # skip = []
    for _ in range(1):
        EMB = Dense(32, activation='linear', kernel_constraint=cn(), kernel_regularizer=rl(),  use_bias=False)(EMB)
        EMB = BatchNormalization(beta_regularizer=rl(), gamma_regularizer=rl())(EMB)
        #EMB = GaussianNoise(0.3)(EMB)
        EMB = Activation("elu")(EMB)
        #EMB = ActivityRegularization(l2=0.01)(EMB)
        EMB = Dropout(0.5)(EMB)





    EMB = Dense(n_neurons, activation='linear', kernel_constraint=cn(),kernel_regularizer=rl(),  use_bias=False)(EMB)
    EMB = BatchNormalization(beta_regularizer=rl(), gamma_regularizer=rl())(EMB)
    EMB = Activation("tanh")(EMB)
    #EMB = ActivityRegularization(l2=0.01)(EMB)
    #EMB = GaussianNoise(0.3)(EMB)


    EMB_model = Model(inputs=EMB_inputs, outputs=[EMB])
    EMB_model.compile(optimizer=get_optimizer(), loss=custom_loss)

    SC_DC_inputs = Input(shape=(n_neurons,))



    sc_layer = Dense(2, activation='linear', kernel_constraint=cn(),  kernel_regularizer=rl(), trainable=True)

    SC = sc_layer(SC_DC_inputs)
    #SC = BatchNormalization()(SC)
    SC = Activation("softmax")(SC)
    SC_model_D = Model(inputs=SC_DC_inputs, outputs=[SC], name="so_model")
    SC_model = Model(inputs=EMB_inputs, outputs=[SC_model_D(EMB)])
    SC_model.compile(optimizer=get_optimizer(), loss={'so_model': custom_loss})

    DC = SC_DC_inputs
    #DC = GaussianNoise(0.5)(DC)

    # DC = Dense(32, activation='linear', kernel_constraint=cn(), use_bias=False)(DC)
    # DC = BatchNormalization()(DC)
    # DC = Activation("elu")(DC)
    # DC = Dropout(0.5)(DC)

    DC = Dense(2, activation='linear', kernel_constraint=cn())(DC)
    #DC = BatchNormalization()(DC)
    DC = Activation("softmax")(DC)
    DC_model_D = Model(inputs=SC_DC_inputs, outputs=[DC], name="do_model")

    DC_model_D.trainable = False
    SCDC_model = Model(inputs=EMB_inputs, outputs=[SC_model_D(EMB), DC_model_D(EMB)])
    SCDC_model.compile(optimizer=a,
                       loss={'so_model': custom_loss, 'do_model': custom_loss},
                       loss_weights={'so_model': 1, 'do_model': do_strength}, )

    DC_model_D.trainable = True
    EMB_model.trainable = False

    DC_model = Model(inputs=EMB_inputs, outputs=[DC_model_D(EMB_model(EMB_inputs))])

    DC_model.compile(optimizer=a,
                     loss={'do_model': custom_loss},
                     loss_weights={'do_model': do_strength}, metrics=['accuracy'])

    return SCDC_model, SC_model, DC_model, EMB_model, DC_model_D  # , l


def train(Xs, ys, Xt, n_iterations, models, enable_dann, batch_size, plot, should_mixup, features):
    #print(features)
    #exit()
    #Xt = Xt.T[features].T
    model, source_classification_model, domain_classification_model, embeddings_model, domain_classification_model_stand_alone = models

    if (should_mixup):
        sample_weights_adversarial = np.ones((batch_size,))
    else:
        sample_weights_adversarial = np.ones((batch_size * 2,))
    # if(MIXUP):
    S_batches = batch_generator([Xs, to_categorical(ys)], batch_size)

    T_batches = batch_generator([Xt, np.zeros(shape=(len(Xt), 2))], batch_size)

    history_training = []
    history_validation = []
    history_validation_weights = []

    log = []
    # print(n_iterations)
    JSD = 0
    activations = []

    found_top = True

    # print(found_top)

    def get_bounded_batch_random(a, b):
        return list((b - a) * np.random.random(batch_size // 2) + a)

    for i in range(n_iterations):

        # print(i)
        # # # print(y_class_dummy.shape, ys.shape)
        if (should_mixup):
            y_adversarial_1 = to_categorical(np.array(([1] * (batch_size // 2) + [0] * (batch_size // 2))))  ###
            y_adversarial_2 = to_categorical(np.array(([0] * (batch_size // 2) + [1] * (batch_size // 2))))  ##
        else:
            y_adversarial_1 = to_categorical(np.array(([1] * (batch_size) + [0] * (batch_size))))  ###
            y_adversarial_2 = to_categorical(np.array(([0] * (len(Xs)) + [1] * (len(Xt)))))  ##

        X0, y0 = next(S_batches)
        X1, _ = next(T_batches)

        # X0 = X0 + np.random.normal(0.0,0.2)
        if (should_mixup):
            X0, y0 = mixup(X0, y0, bs=batch_size // 2)
            y1 = source_classification_model.predict(X1, batch_size=1000)
            X1, y1 = mixup(X1, y1, bs=batch_size // 2)

        X_adv = np.concatenate([Xs, Xt])
        y_class = np.concatenate([y0, np.zeros_like(y0)])



        sample_weights_class = np.concatenate([np.ones(shape=len(y0)),
                                               np.zeros(shape=len(y0))])

        # print(y_class.shape, y_adversarial_1.shape, sample_weights_adversarial.shape, sample_weights_class.shape)

        if (enable_dann):
            # print(X_adv.shape, y_class.shape, y_adversarial_1.shape, sample_weights_class.shape, sample_weights_adversarial.shape)
            # note - even though we save and append weights, the batchnorms moving means and variances
            # are not saved throught this mechanism
            for _ in range(0,1):
                domain_classification_model.train_on_batch(X_adv, [y_adversarial_2])
            # sc = domain_classification_model.predict(X_adv)
            # domain_score = accuracy_score((y_adversarial_2.argmax(axis = 1)), sc.argmax(axis=1))
            # print(domain_score)

            # X_adv = X_adv + noise
            stats = model.train_on_batch(np.concatenate([X0,X1]), [y_class, y_adversarial_1],
                                         sample_weight=[sample_weights_class, sample_weights_adversarial])

            # sharpening
            # # r_Xt = X1  # + noise
            # y_t_hat = source_classification_model.predict(Xt, batch_size=1000)
            #
            # y_t_hat = to_categorical(np.argmax(y_t_hat, axis=1), num_classes=2)
            #
            # source_classification_model.train_on_batch(Xt, y_t_hat)



            # if (training_score > 0.999):
            #     break
            #
            # if(i%100 ==0):
            #     embsXs = embeddings_model.predict(Xs).T[0].mean()
            #     embsXt = embeddings_model.predict(Xt).T[0].mean()
            #
            #     training_predict = (source_classification_model.predict(Xs, verbose=False))
            #     training_score = accuracy_score((ys), training_predict.argmax(axis=1))
            #     print(i, training_score,embsXs, embsXt)

        else:
            (source_classification_model.train_on_batch(X0, [y0]))
            training_predict = (source_classification_model.predict(Xs, verbose=False))
            training_score = accuracy_score((ys), training_predict.argmax(axis=1))
            # print(training_score)
            # if (i % 1000 == 0):
            #     print(i, training_score)

            if (training_score > 0.999):
                found_top = True

                # if (plot):
                #     training_predict = (source_classification_model.predict(Xs, verbose=False))
                #     validation_predict = (source_classification_model.predict(Xv, verbose=False))
                #     training_score = accuracy_score((ys), training_predict.argmax(axis=1))
                #     validation_score = accuracy_score((yv), validation_predict.argmax(axis=1))
                #     # print(training_score)
                #
                #     embsXs = embeddings_model.predict(Xs)
                #     embsXt = embeddings_model.predict(Xt)
                #
                #     Xs_p, Xt_p, = source_classification_model.predict(Xs).argmax(axis=1), source_classification_model.predict(
                #         Xt).argmax(axis=1)
                #
                #     WMD = []
                #     for j in range(len(embsXs.T)):
                #         WMD += [wsd(embsXs[:, j], embsXt[:, j])]
                #     WMD = np.array(WMD).mean()
                #
                #     # JSD = wasser(embsXs, embsXt)
                #     if (i % 100 == 0):
                #         # print(embsXt)
                #         print(i, WMD, JSD, enable_dann, training_score, stats, sum(Xs_p), sum(Xt_p), training_score,
                #               validation_score, "embs", embsXs.T[0].round().mean(), embsXt.T[0].round().mean())
                #         # print(training_predict)
                #     # print(sum(Xs_p),  (sum(ys)), sum(training_predict.argmax(axis=1)))
                #     # print(i,WMD,JSD, training_score)
                #     log.append([i, WMD, "WMD"])
                #     log.append([i, JSD, "WSD"])
                #     log.append([i, training_score, "Accuracy score"])
                #     activations.append([embsXs, embsXt, ys, yv])
                #

    if (plot):
        dump_csv("data-rand.csv", log, ["Iteration", "Metric", "Type"])
        import pickle
        with open('activations', 'wb') as fp:
            pickle.dump(activations, fp)
    embsXs = embeddings_model.predict(Xs).T[0].mean()
    embsXt = embeddings_model.predict(Xt).T[0].mean()

    # eXs = embeddings_model.predict(Xs)
    # eXt = embeddings_model.predict(Xt)
    #
    # # Plot Data
    # plt.scatter(eXs[:, 0], eXs[:, 1], marker='o', c=[["red", "blue"][k] for k in ys], alpha=0.4)
    # plt.scatter(eXt[:, 0], eXt[:, 1], marker='x', c=["grey"], alpha=0.8)
    # plt.show()
    # print(embsXs, embsXt)

    return None  #
