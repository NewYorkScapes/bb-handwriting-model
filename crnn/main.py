import numpy as np
from Data_Helper import *
from crnn_helper import * 
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from datetime import datetime


args = parse_argument()
final_report = ''

###### Data loading/ Train Test Spliting/ image Loading / alphabet definition / Model parameter definition ###### 
print('###### Data loading/ Train Test Spliting/ image Loading / alphabet definition / Model parameter definition ###### ')
print()

# load data 
main_path = '/scratch/nmw2/bb-handwriting-model/'
path_to_seg_csv = main_path + 'data/' + args.trainedfilename
path_to_bb_segs = main_path + 'data/Transcribed_Segs'
path_to_iam_csv = main_path + 'data/iam_train_subset.csv'
path_to_iam_segs = main_path + 'data/iam_train_subset'

df = load_data(args.runtype, path_to_seg_csv,path_to_bb_segs,path_to_iam_csv = path_to_iam_csv)

sys.stdout.write('Data loaded successfully')

# get transcription stats: optional
print()
stats = get_stats_discrepancy(path_to_seg_csv)
print()

# train test split
X_train, X_val, y_train, y_val = data_train_val_split(df, test_size= 0.2 ,IAM_USED= True)
sys.stdout.write('Train/test split loaded successfully\n')
# reset index and get size of train/val 
X_train.reset_index(inplace = True, drop=True) 
y_train.reset_index(inplace = True, drop=True)
X_val.reset_index(inplace = True, drop=True) 
y_val.reset_index(inplace = True, drop=True)
train_size = X_train.size
valid_size= X_val.size


# load images for train/val set
train_imgs = load_imgs(X_train, path_to_bb_segs,path_to_iam_segs, max_height= 64, max_width= 256)
val_imgs = load_imgs(X_val, path_to_bb_segs,path_to_iam_segs, max_height= 64, max_width= 256)
sys.stdout.write('Image loading successfully')
sys.stdout.write('\n')
train_imgs = np.array(train_imgs)
val_imgs = np.array(val_imgs)

# make alphabet
# here it includes all possible alphabets
# true means preset alphatbet + any additional labels appearing in dataset 
# false means no preset alphatbet (decrease alphabet size might improve model prediction accuracy)
alphabets = get_alphabets(y_train, y_val, True)

# define parameters for the length of input labels 
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels

# make train_y, train_label_len, train_input_len, train_output which are necessary for the CNN/RNN to understand the constraint of the model
train_y, train_label_len, train_input_len, train_output = label_helper(max_str_len, num_of_characters, num_of_timestamps, train_size, y_train, alphabets)
valid_y, valid_label_len, valid_input_len, valid_output = label_helper(max_str_len, num_of_characters, num_of_timestamps, valid_size, y_val, alphabets)

sys.stdout.write('Total number of alphatbet {}'.format(len(alphabets)))
sys.stdout.write('\n')



print()
####### Define Model #############
print(('####### Define Model #############'))

input_data = Input(shape=(256, 64, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.3)(inner)

# CNN to RNN
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

## RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

## OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
print(model.summary())

# define labels, input_length and label_length
labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# define ctc_loss and wrap up model definitions (model_final )
ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)


############ RUN MODEL #########################
print('############ RUN MODEL #########################')
print()
# define model parameters
lr = 0.0001
epoch_size = int(args.numepochs)
batch_size = 64

final_report = final_report + 'Number of epochs: ' + args.numepochs + '\n' + 'Batch size: ' + str(batch_size) + '\n'

# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = lr))

model_final.fit(x=[train_imgs, train_y, train_input_len, train_label_len], y=train_output, 
                validation_data=([val_imgs, valid_y, valid_input_len, valid_label_len], valid_output),
                epochs=epoch_size, batch_size=batch_size)


# Make predictions for validation set and show results.
prediction = get_prediction(model,val_imgs,alphabets)
correct_info, final_report = get_prediction_accuracy(prediction, y_val, X_val, final_report)

if len(correct_info):
    final_report = final_report + 'List of correctly predicted words: \n'
    for i in correct_info:
        print('predicted: ', correct_info[i]['predicted'], '| label: ', correct_info[i]['label'])
        final_report = final_report + 'predicted: ' + correct_info[i]['predicted'] + '| label: ' + correct_info[i]['label'] +  '\n'

# Save model

cur_date = datetime.today().strftime('%Y-%m-%d')
model.save('models/' + cur_date + '_' + args.runtype + '_model')
with open('run_summaries.txt', 'a') as f:
    f.write(cur_date + '\n')
    f.write('Run type: ' + args.runtype + '\n'  )
    f.write(final_report + '----------------\n\n')
    f.close()
