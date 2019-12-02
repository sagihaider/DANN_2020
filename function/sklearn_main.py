import os
import time

import numpy as np
from keras import backend as K
from scipy.stats import sem
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from utils import entropy1
import scipy.io as spio
from numpy import zeros
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn_nn import NNClassifier
#from sklearn_nn_evo import NNClassifier
from bagging import BaggingClassifierDomainAdaptation as BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
#from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
###
N_EPOCHS = 2000


N_THREADS = 1

K.set_session(K.tf.Session(
    config=K.tf.ConfigProto(intra_op_parallelism_threads=N_THREADS, inter_op_parallelism_threads=N_THREADS)))


parser = ArgumentParser()
parser.add_argument("-s", "--subject", dest="subject",
                    help="subject number", metavar="int")

args = parser.parse_args()

cols = 2
rows = 9
results = zeros([rows, cols])
results_dann = zeros([rows, cols])

subject = args.subject
fName = './EEG_Features/A0' + str(subject) + '_FBCSP.mat'  # Load Data
print(fName)
mat = spio.loadmat(fName)
Xs = mat['Train_X']
Xt = mat['Test_X']
a = int(len(Xs) / 2)
b = int(len(Xt) / 2)
ys = np.concatenate([np.zeros(a, dtype=int), np.ones(a, dtype=int)])
yt = np.concatenate([np.zeros(b, dtype=int), np.ones(b, dtype=int)])


#
# ss = StandardScaler()
# ss.fit(np.concatenate([Xs,Xt]))
# Xs = ss.transform(Xs)
# Xt = ss.transform(Xt)



clf = BaggingClassifier(NNClassifier(N_EPOCHS, Xt, enable_dann=True, batch_size=64), n_estimators=2000, bootstrap_features=False, bootstrap=False, verbose=1000)
#clf = AdaBoostClassifier(NNClassifier(N_EPOCHS, Xt, enable_dann=True),  n_estimators=10,  learning_rate=0.2)


#clf = Pipeline([  ('nn', clf)])
#('minmax', StandardScaler()),

#clf = NNClassifier(N_EPOCHS, Xt, enable_dann=True)
clf.fit(Xs, ys)
#
# embsXs = []
# embsXt = []
# for model in clf.estimators_:
#     embsXs.append(model.predict_embs(Xs))
#     embsXt.append(model.predict_embs(Xt))
#
# Xs = np.concatenate(embsXs, axis = -1)
# Xt = np.concatenate(embsXt, axis = -1)
#clf = XGBClassifier(n_jobs=6)

# Xs = np.concatenate([np.random.normal(0,0.3, size = Xs.shape) + Xs for _ in range(10)], axis = 0)
#
# ys = np.concatenate([ys  for _ in range(10)], axis = -1)

# clf = RandomForestClassifier(n_jobs=6, n_estimators=10000)
# clf.fit(Xs,ys)
#
#
# #print("OOB", subject, clf.oob_score_)
# #print(clf.estimator_weights_ )
#
ys_hat = clf.predict(Xs)
yt_hat = clf.predict(Xt)
s_acc = accuracy_score(ys, ys_hat)
t_acc = accuracy_score(yt, yt_hat)
t2_acc = accuracy_score(yt, clf.predict(Xt + np.random.normal(0,0.3, size = Xt.shape)))


print("\033[1;32;40m ")
print("Subject Accuracy", subject, s_acc, t_acc, t2_acc)
print("\033[0;37;40m")




