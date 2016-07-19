import pandas as pd
import numpy as np
import chainer
from chainer import Function, Variable, optimizers, cuda
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=1000, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=650, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=20,
                    help='learning minibatch size')
parser.add_argument('--bproplen', '-l', type=int, default=35,
                    help='length of truncated BPTT')
parser.add_argument('--gradclip', '-c', type=int, default=5,
                    help='gradient norm threshold to clip')
parser.add_argument('--validation', '-v', type=int, default=100,
                    help='wide of validation')


args = parser.parse_args()
np.random.seed(0)

xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch   # number of epochs
n_units = args.unit  # number of units per layer
batchsize = args.batchsize   # minibatch size
bprop_len = args.bproplen   # length of truncated BPTT
grad_clip = args.gradclip    # gradient norm threshold to clip
validation = args.validation



TEMPERATURE_TRAIN_FEATURE_PATH = 'Temperature_Train_Feature.tsv'
data_train_feature_tem = pd.read_csv(TEMPERATURE_TRAIN_FEATURE_PATH, sep='\t')

PRECIPITATION_TRAIN_FEATURE_PATH = 'Precipitation_Train_Feature.tsv'
data_train_feature_pre = pd.read_csv(PRECIPITATION_TRAIN_FEATURE_PATH, sep='\t')

SUNDURATION_TRAIN_FEATURE_PATH = 'SunDuration_Train_Feature.tsv'
data_train_feature_sun = pd.read_csv(SUNDURATION_TRAIN_FEATURE_PATH, sep='\t')

X_tem = data_train_feature_tem.loc[:, ['place%d' % i for i in range(11)]].values
X_loc = data_train_feature_tem.loc[:, ['targetplaceid']].values
X_date = data_train_feature_tem.loc[:, ['day','hour' ]].values
X_pre = data_train_feature_pre.loc[:, ['place%d' % i for i in range(11)]].values
X_sun = data_train_feature_sun.loc[:, ['place%d' % i for i in range(11)]].values

X = np.hstack((X_tem, X_date, X_loc, X_pre, X_sun))
#X[:,11] /= sum(X[:,11]) 
#X[:,12] /= sum(X[:,12]) 
#X[:,13] /= sum(X[:,13]) 

TEMPERATURE_TRAIN_TARGET_PATH = 'Temperature_Train_Target.dat.tsv'
y = np.loadtxt(TEMPERATURE_TRAIN_TARGET_PATH).reshape((1800, 1))

from sklearn.cross_validation import train_test_split
TEST_SIZE = 0.2
RANDOM_STATE = 0
X_train, X_val = np.vsplit(X,[1440])
y_train, y_val = np.vsplit(y,[1440])

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
'''
print X_train.shape
print X_val.shape
print y_train.shape
print  y_val.shape
sys.exit()
'''
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean', axis=0)
imp.fit(X_train)
X_train = imp.transform(X_train)
X_val = imp.transform(X_val)



TEMPERATURE_TEST_FEATURE_PATH = 'Temperature_Test_Feature.tsv'
data_test_feature_tem = pd.read_csv(TEMPERATURE_TEST_FEATURE_PATH, sep='\t')

PRECIPITATION_TEST_FEATURE_PATH = 'Precipitation_Test_Feature.tsv'
data_test_feature_pre = pd.read_csv(PRECIPITATION_TEST_FEATURE_PATH, sep='\t')

SUNDURATION_TEST_FEATURE_PATH = 'SunDuration_Test_Feature.tsv'
data_test_feature_sun = pd.read_csv(SUNDURATION_TEST_FEATURE_PATH, sep='\t')

X_test_tem = data_test_feature_tem.loc[:, ['place%d' % i for i in range(11)]].values
X_test_loc = data_test_feature_tem.loc[:, ['targetplaceid']].values
X_test_date = data_test_feature_tem.loc[:, ['day','hour' ]].values
X_test_pre = data_test_feature_pre.loc[:, ['place%d' % i for i in range(11)]].values
X_test_sun = data_test_feature_sun.loc[:, ['place%d' % i for i in range(11)]].values

X_test = np.hstack((X_test_tem, X_test_date, X_test_loc, X_test_pre, X_test_sun))
#X_test[:,11] /= sum(X_test[:,11]) 
#X_test[:,12] /= sum(X_test[:,12]) 
#X_test[:,13] /= sum(X_test[:,13]) 

X_test = np.array(X_test, dtype=np.float32)
X_test = imp.transform(X_test)



class LSTM(Chain):
    def __init__(self, n_units):
        super(LSTM, self).__init__(
            l1=L.Linear(36, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.LSTM(n_units, n_units),
            l4=L.LSTM(n_units, n_units),
            l5=L.Linear(n_units, 1)
        )

    def __call__(self, x):
        h_1 = self.l1(x)
        h_2 = self.l2(h_1)
        h_3 = self.l3(h_2)
        h_4 = self.l4(h_3)
        o = self.l5(h_4)
        return o

    def reset_state(self):
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()   

model = LSTM(n_units)
model.compute_accuracy = False 

if args.gpu >= 0:
    model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

whole_len = 1440
accum_loss = 0
perp = 0
epoch = 0
jump = whole_len // batchsize
batch_idxs = list(range(batchsize))

for i in range(jump * n_epoch):
    x = Variable(xp.asarray([X_train[(jump * j + i) % whole_len] for j in batch_idxs]))
    t = Variable(xp.asarray([y_train[(jump * j + i) % whole_len] for j in batch_idxs]))
    loss_i = F.mean_squared_error(model(x), t)
    accum_loss += loss_i
    perp += loss_i.data

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        model.zerograds()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = 0
        optimizer.update()

    if (i + 1) % (jump*validation) == 0:
        epoch += validation
        print 'epoch:%d'%epoch
        print 'loss:%f'%(perp / (jump*validation))
        perp = 0


        model_val = model.copy()
        model_val.reset_state()
        x_val = X_val
        t_val = xp.asarray(y_val)
        y_val_pred = xp.empty([360, 1], dtype=xp.float32)
        for j in range(360):
            tmp = xp.asarray(x_val[j].reshape((1,36)))
            y_val_pred[j][0] = model_val(Variable(tmp, volatile='on')).data    
        loss = F.mean_squared_error(Variable(y_val_pred, volatile='on'), Variable(t_val, volatile='on'))
        print 'validation:%f'%loss.data

    '''
    if (i + 1) % 10000000 == 0:
        model_test = model.copy()
        model_test.reset_state()
        x_test = X_test
        y_test_pred = xp.empty([1800, 1], dtype=xp.float32)
        for j in range(1800):
            tmp = xp.asarray(x_test[j].reshape((1,36)))
            y_test_pred[j][0] = model_test(Variable(tmp), False).data 
        SUBMIT_PATH = 'submission_cpu_input_dif_drop%d.dat'% ((i + 1) / jump)
        pred = cuda.to_cpu(y_test_pred)
        np.savetxt(SUBMIT_PATH, pred, fmt='%.10f')
    '''



'''
model.reset_state()
x = X_val
t = xp.asarray(y_val)
model.zerograds()
y_test_pred = xp.empty([360, 1], dtype=xp.float32)
for j in range(batchsize):
	tmp = xp.asarray(x[j].reshape((1,36)))
	y_test_pred[j][0] = model(Variable(tmp)).data
loss = F.mean_squared_error(Variable(y_test_pred), Variable(t))
print loss.data
'''



'''
model.reset_state()
y_test_pred = model(x_test)
print y_test_pred.data.shape
SUBMIT_PATH = 'submission.dat'
np.savetxt(SUBMIT_PATH, y_test_pred.data, fmt='%.10f')
'''