import pandas as pd
import numpy as np
import chainer
from chainer import Function, Variable, optimizers, cuda, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import argparse
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.utils import resample, shuffle
from xgboost import XGBRegressor 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=10000, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=50, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=29,
                    help='learning minibatch size')
parser.add_argument('--bproplen', '-bp', type=int, default=35,
                    help='length of truncated BPTT')
parser.add_argument('--evaluation', '-v', type=int, default=10,
                    help='frequency of evaluation')
parser.add_argument('--end_counter_max', '-ec', type=int, default=10,
                    help='maximum of end_counter')
parser.add_argument('--dropout', '-d', type=int, default=0,
                    help='wiyh or without dropout')
parser.add_argument('--dnn_number', '-dn', type=int, default=6,
                    help='the number of dnn')
parser.add_argument('--xgb_number', '-xn', type=int, default=3,
                    help='the number of xgb')
args = parser.parse_args()

n_epoch = args.epoch   # number of epochs
n_units = args.unit  # number of units per layer
batchsize = args.batchsize   # minibatch size
bprop_len = args.bproplen   # length of truncated BPTT
evaluation = args.evaluation
end_counter_max = args.end_counter_max
dropout = args.dropout
dnn_number = args.dnn_number
xgb_number = args.xgb_number

xp = cuda.cupy if args.gpu >= 0 else np


#Data Construction
data_tem = pd.read_csv('data/Temperature.tsv', sep='\t')
data_pre = pd.read_csv('data/Precipitation.tsv', sep='\t')
data_sun = pd.read_csv('data/SunDuration.tsv', sep='\t')
x_date = data_tem.loc[:, ['day','hour' ]].values
x_loc = data_tem.loc[:, ['targetplaceid']].values
x_tem = data_tem.loc[:, ['place%d' % i for i in range(11)]].values
x_pre = data_pre.loc[:, ['place%d' % i for i in range(11)]].values
x_sun = data_sun.loc[:, ['place%d' % i for i in range(11)]].values
x = np.hstack((x_date, x_loc, x_tem, x_pre, x_sun)) #numpy
#x[:,0] /= sum(X[:,0]) 
#x[:,1] /= sum(X[:,1]) 
#x[:,2] /= sum(X[:,2]) 
y = np.loadtxt('data/Temperature_Target.tsv').reshape(1800, 1) #numpy

imp = Imputer(strategy='mean', axis=0)
imp.fit(x)
x = imp.transform(x)
x = np.array(x, dtype=np.float32)
imp.fit(y)
y = imp.transform(y)
y = np.array(y, dtype=np.float32)


class Deep(Chain):
    def __init__(self):
        super(Deep, self).__init__(
            l1=L.Linear(36,10),
            l2=L.Linear(10, 5),
            l3=L.Linear(5, 1)
        )
        
    def __call__(self, x):
        h_1 = F.relu(self.l1(x))
        h_2 = F.relu(self.l2(h_1))
        o = self.l3(h_2)
        return o   
    
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
        out = self.l5(h_4)
        return out

    def reset_state(self):
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()

class LSTM_dropout(Chain):
    def __init__(self, n_units):
        super(LSTM_dropout, self).__init__(
            l1=L.Linear(36, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.LSTM(n_units, n_units),
            l4=L.LSTM(n_units, n_units),
            l5=L.Linear(n_units, 1)
        )

    def __call__(self, x, train):
        h_1 = self.l1(x)
        h_2 = self.l2(F.dropout(h_1, ratio=0.5, train = train))
        h_3 = self.l3(F.dropout(h_2, ratio=0.5, train = train))
        h_4 = self.l4(F.dropout(h_3, ratio=0.5, train = train))
        out = self.l5(h_4)
        return out

    def reset_state(self):
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()


class Learning_model():
    def DNN(self, x_train, y_train, x_test, y_test, seed):
        np.random.seed(seed)
        dnn = Deep()
        dnn.compute_accuracy = False

        if args.gpu >= 0:
            dnn.to_gpu()

        optimizer = optimizers.Adam()
        optimizer.setup(dnn)

        end_counter = 0
        min_loss = 100
        final_epoch = 0
        final_pred = xp.empty([x_test.shape[0], 1], dtype=xp.float32)

        x_train, y_train = resample(x_train, y_train, n_samples=x_train.shape[0])
        for epoch in range(n_epoch):
            indexes = np.random.permutation(x_train.shape[0])
            for i in range(0, x_train.shape[0], batchsize):
                x_train_dnn = Variable(x_train[indexes[i : i + batchsize]])
                y_train_dnn = Variable(y_train[indexes[i : i + batchsize]])
            dnn.zerograds()
            loss = F.mean_squared_error(dnn(x_train_dnn), y_train_dnn)
            loss.backward()
            optimizer.update()
            end_counter += 1
        
            #evaluation
            if epoch % evaluation == 0:
                y_pred = dnn(Variable(x_test, volatile='on'))
                loss = F.mean_squared_error(y_pred, Variable(y_test, volatile='on'))

                if min_loss > loss.data:
                    min_loss = loss.data
                    print "epoch{}".format(epoch)
                    print "Current minimum loss is {}".format(min_loss)
                    serializers.save_npz('network/DNN{}.model'.format(seed), dnn)
                    final_epoch = epoch
                    final_pred = y_pred
                    end_counter = 0

            if end_counter > end_counter_max:
                f = open("network/final_epoch.txt", "a")
                f.write("DNN{}:{}".format(seed, final_epoch) + "\n")
                f.close()
                break     

        return final_pred.data, min_loss    

    def RNN(self, x_train, y_train, x_test, y_test, seed):
        np.random.seed(seed)
        if dropout == 1:
            lstm = LSTM_dropout(n_units)
        else:
            lstm = LSTM(n_units)
        lstm.compute_accuracy = False 

        if args.gpu >= 0:
            lstm.to_gpu()

        optimizer = optimizers.Adam()
        optimizer.setup(lstm)

        whole_len = x_train.shape[0]
        jump = whole_len // batchsize
        batch_idxs = list(range(batchsize))
        epoch = 0
        lstm_loss = 0
        accum_loss = 0
        min_loss = 1
        end_counter = 0
        final_epoch = 0
        final_pred = xp.empty([x_test.shape[0], 1], dtype=xp.float32)

        for i in range(jump * n_epoch):
            x_train_lstm = Variable(xp.asarray([x_train[(jump * j + i) % whole_len] for j in batch_idxs]))
            y_test_lstm = Variable(xp.asarray([y_train[(jump * j + i) % whole_len] for j in batch_idxs]))
            if dropout == 1:
                loss = F.mean_squared_error(lstm(x_train_lstm, True), y_test_lstm)
            else:
                loss = F.mean_squared_error(lstm(x_train_lstm), y_test_lstm)
            lstm_loss += loss
            accum_loss += loss.data

            #truncated BP
            if (i + 1) % bprop_len == 0:  # Run truncated BPTT
                lstm.zerograds()
                lstm_loss.backward()
                lstm_loss.unchain_backward()  # truncate
                lstm_loss = 0
                optimizer.update()

            #evaluation
            if (i + 1) % (jump*evaluation) == 0:
                epoch += evaluation
                #print 'loss:{}'.format(accum_loss / (jump*evaluation))
                accum_loss = 0

                lstm_eval = lstm.copy()
                lstm_eval.reset_state()
                x_test_lstm = xp.asarray(x_test)
                y_test_lstm = xp.asarray(y_test)
                y_pred = xp.empty([x_test.shape[0], 1], dtype=xp.float32)
                for j in range(x_test_lstm.shape[0]):
                    one = Variable(x_test_lstm[j].reshape((1,36)), volatile='on')
                    if dropout == 1:
                        y_pred[j][0] = lstm_eval(one, False).data   
                    else:
                        y_pred[j][0] = lstm_eval(one).data
                loss = F.mean_squared_error(Variable(y_pred, volatile='on'), Variable(y_test_lstm, volatile='on'))
                end_counter += 1
                #print 'evaluation:{}'.format(loss.data)

                #save the best model
                if min_loss > loss.data:
                    min_loss = loss.data
                    print 'epoch:{}'.format(epoch)
                    print "Current minimum loss is {}".format(min_loss)
                    serializers.save_npz('network/LSTM{}.model'.format(seed), lstm)
                    final_epoch = epoch
                    final_pred = y_pred
                    end_counter = 0

            if end_counter > end_counter_max:
            	f = open("network/final_epoch.txt", "a")
            	f.write("LSTM{}:{}".format(seed, final_epoch) + "\n")
            	f.close()
            	break

        return final_pred.data, min_loss       


    def XGB(self, x_train, y_train, x_test, y_test):
        x_train, y_train = shuffle(x_train, y_train)
        xgb = XGBRegressor(max_depth=4, subsample=0.9)
        xgb.fit(x_train,y_train)
        y_pred = xgb.predict(x_test).reshape(x_test.shape[0], 1)
        loss = mean_squared_error(y_pred, y_test)
        print loss
        return y_pred, loss


x_train, x_test = np.vsplit(x,[1440])
y_train, y_test = np.vsplit(y,[1440])

#bagging
p = []
l = []
model = Learning_model()
for i in range(dnn_number):
    p_i, l_i = model.DNN(x_train, y_train, x_test, y_test, i)
    p.append(p_i)
    l.append(l_i)
l_ave = sum(l)/dnn_number
pred = sum(p)/dnn_number
print "Single average loss:{}".format(l_ave)
print "Bagging loss:{}".format(mean_squared_error(pred, y_test))

p_xgb = []
l_xgb = []
for i in range(xgb_number):
    p_i, l_i = model.XGB(x_train, y_train, x_test, y_test)
    p_xgb.append(p_i)
    l_xgb.append(l_i)
l_ave_xgb = sum(l_xgb)/xgb_number
pred_xgb = sum(p_xgb)/xgb_number
print "XGB single average loss:{}".format(l_ave_xgb)
print "XGB bagging loss:{}".format(mean_squared_error(pred_xgb, y_test))

for w in range(101):
    en_pred = (1-w/100.0)*pred + w/100.0*pred_xgb
    print "DNN({}),XGB({}):{}".format(1-w/100.0, w/100.0, mean_squared_error(en_pred, y_test))
