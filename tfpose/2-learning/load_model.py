import pandas as pd
import argparse
import numpy as np
from tensorflow import keras

parser = argparse.ArgumentParser(description='for load models...')
parser.add_argument('--input', type=str, required=True, help='input of dnn from data_prepared (pickle)')
parser.add_argument('--model', type=str, required=True, help='path of model')
args = parser.parse_args()

#calculate accuracy and build predcition&label set
def cal_accuracy(test_labels, test_predictions):
    answer = test_labels.to_numpy()
    pl_set = np.concatenate((np.reshape(test_predictions, (test_predictions.shape[0], 1)), np.reshape(answer, (answer.shape[0], 1))) , axis=1)

    standard = 0.5
    true_cnt = 0
    for pl in pl_set:
        if pl[0] <= standard and pl[1] == 0:
            true_cnt += 1
        elif pl[0] > standard and pl[1] == 1:
            true_cnt += 1

    accuracy = true_cnt / len(test_labels) * 100
    return pl_set, accuracy

##main
#read model
model = keras.models.load_model(args.model)
#model.summary()

#read pickle data (time-flow, nonshuffled)
test_dataset = pd.read_pickle('../0-data/data_prepared/'+args.input)
test_labels = test_dataset.pop('label')

#calculate prediction
test_predictions = model.predict(test_dataset).flatten()
pl_set, accuracy = cal_accuracy(test_labels, test_predictions)

#prediction to pickle
pl_df = pd.DataFrame(pl_set, columns=['prediction','label'])
#print(pl_df)
pl_df.to_pickle('../0-data/data_prediction/kjk2kjk1_'+str(accuracy//0.01/100)+'.pkl')

print((pl_set[:-50]))
print("accuracy: ",accuracy,"%")