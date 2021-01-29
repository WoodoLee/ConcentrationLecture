import pathlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout

parser = argparse.ArgumentParser(description='for running DNN...')
parser.add_argument('--file', type=str, required=True, help='name of file')
parser.add_argument('--plot', type=str, default='0', help='choose plot graph')
parser.add_argument('--size', type=int, default=0, help='choose size of dataset')
parser.add_argument('--epoch', type=int, default=10000, help='number of iterations')
args = parser.parse_args()

#build dataset
def build_dataset(cnt):
    rawdata = pd.read_pickle("../0-data/data_prepared/" + args.file)
    if cnt > len(rawdata):
        return pd.DataFrame()
    elif cnt == 0:
        return rawdata
    else:
        return rawdata[:cnt]
#DNN model

def build_model():
    model = keras.Sequential([
        #layers.Dense(len(train_dataset.keys()), activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(8, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(8, activation = 'relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    #keras.optimizers.RMSprop(0.1)
    keras.optimizers.Adam(lr=0.001)
    #keras.optimizers.Adagrad(lr=0.001)
    #keras.optimizers.Adadelta(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer = 'Adam', metrics = ['binary_crossentropy'])
    #metrics = ['mae', 'mse','accuracy'])
    return model

#prints dot('.') for every epoch for visual convenience
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

#calculate accuracy and build predcition&label set
def cal_accuracy(test_labels, test_predictions):
    answer = test_labels.to_numpy()
    pl_set = np.concatenate((np.reshape(test_predictions, (test_predictions.shape[0], 1)), np.reshape(answer, (answer.shape[0], 1))) , axis=1)
    '''
    #cal standard
    sorted_predictions = np.sort(pl_set)
    standard_idx = np.sum(test_labels)-1
    standard = sorted_predictions[standard_idx,0]
    #print(standard)
    '''
    standard = 0.5
    true_cnt = 0
    for pl in pl_set:
        if pl[0] <= standard and pl[1] == 0:
            true_cnt += 1
        elif pl[0] > standard and pl[1] == 1:
            true_cnt += 1

    accuracy = true_cnt / len(test_labels) * 100
    return pl_set, accuracy

#Graph of epoch vs. binary_crossentropy for trainset, val_set 
def plot_history1(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    '''
    '''
    #font for ieee
    fontpath = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    prop = fm.FontProperties(fname=fontpath)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams.update({'font.size':40})
    

    plt.xlabel('Epoch')
    plt.ylabel('binary_crossentropy')
    plt.scatter(hist['epoch'], hist['binary_crossentropy'], label='Train Error', c= 'blue', alpha=.5)
    plt.scatter(hist['epoch'], hist['val_binary_crossentropy'], label = 'Validation Error', c='red',alpha=.5)
    plt.grid(True)
    plt.title('Model Training')
    plt.ylabel('Binary Cross-entropy')
    plt.legend()
    plt.show()

#Test prediction histogram of label 1, 0
def plot_history2(test_labels, test_predictions, model):
    test_labels_ = np.array(test_labels)
    test_data = np.concatenate((test_predictions.reshape(-1,1), test_labels_.reshape(-1,1)),axis=1)
    test_data = pd.DataFrame({'test_predictions':test_data[:,0], 'test_labels':test_data[:,1]})
    df1 = test_data[test_data['test_labels']==1]
    df0 = test_data[test_data['test_labels']==0]

    fig, axes = plt.subplots(1,2)

    df1.hist('test_predictions', bins=100, ax=axes[1])
    df0.hist('test_predictions', bins=100, ax=axes[0])

    axes[0].set_title("Label 0")
    axes[0].set_xlabel("test_predictions")
    axes[1].set_title("Label 1")
    axes[1].set_xlabel("test_predictions")

    plt.show()

#(optional)Visualizes error distribution using histogram
def plot_history3(test_predictions, test_labels, hist):
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _=plt.ylabel("Count")
    plt.show()


##main
#split test, train set(X: train_dataset)
dataset = build_dataset(args.size)
X = dataset.sample(frac=0.8, random_state = 42)
test_dataset = dataset.drop(X.index)

#split label(y: train_label)
y = X.pop('label')
test_labels = test_dataset.pop('label')

# Stratified k-fold
skf = StratifiedKFold(n_splits=5, random_state=42)

#deep learning for each k-folded set
accuracy = []
i=0

for train_idx, val_idx in skf.split(X, y): 
    train_dataset, val_dataset = X.iloc[train_idx], X.iloc[val_idx]
    train_labels, val_labels = y.iloc[train_idx], y.iloc[val_idx]

    #run dnn
    model = build_model()
    #print(model.summary())

    #non use early stopping
    #history = model.fit(train_dataset, train_labels, epochs=args.epoch, validation_data=(val_dataset, val_labels), verbose=0, callbacks=[PrintDot()])
    
    #early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    history = model.fit(train_dataset, train_labels, epochs=args.epoch, validation_data=(val_dataset, val_labels), verbose=0, callbacks=[PrintDot(), es])
    
    #calculate prediction
    test_predictions = model.predict(test_dataset).flatten()
    pl_set, tmp_accuracy = cal_accuracy(test_labels, test_predictions)
    accuracy.append(tmp_accuracy)
    print(pl_set[-100:])
    print("accuracy: ",tmp_accuracy,"%")

    #save model
    model.save('models/4R16R1Skjk_'+str(i)+'.h5')

    #extract pl_set to pickle
    pl_df = pd.DataFrame(pl_set, columns=['prediction','label'])
    #print(pl_df)
    pl_df.to_pickle('../0-data/data_prediction/'+args.file[:-4]+'('+str(i)+')_'+str(tmp_accuracy//0.01/100)+'.pkl')

    #visualize graph
    hist = pd.DataFrame(history.history)
    print(hist)
    hist['epoch']= history.epoch
    plot = list(map(int, args.plot.split(',')))
    for idx in plot:
        if idx == 1:
            plot_history1(history)
        elif idx == 2:
            plot_history2(test_labels, test_predictions, model)
        elif idx == 3:
            plot_history3(test_predictions, test_labels, hist)
    i+=1

#print accuracy for each fold
print("accuracy for each fold")
print(accuracy)
print("average accuracy: ",sum(accuracy)/len(accuracy),"%")