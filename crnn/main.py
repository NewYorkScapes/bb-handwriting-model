from data_Helper import *
from crnn_helper import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate,Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import matplotlib.pyplot as plt
import random


args = parse_argument()
final_report = ''

###### Data loading/ Train Test Spliting/ image Loading / alphabet definition / Model parameter definition ######
print('>> THIS IS A ', args.runtype.upper(), ' RUN <<\n\n')
print('###### Data loading/ Train Test Spliting/ image Loading / alphabet definition / Model parameter definition ###### ')
print()

# load data 
main_path = '/scratch/yw3076/bb-handwriting-model/'
path_to_seg_csv = main_path + 'data/' + args.trainedfilename
path_to_bb_segs = main_path + 'data/Transcribed_Segs'

df = load_data(args.runtype, path_to_seg_csv,path_to_bb_segs)

# get transcription stats: optional
print()
stats = get_stats_discrepancy(path_to_seg_csv)
print()

# train/val/test split
X_train, X_test, y_train, y_test = data_train_val_split(df, test_size= 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
sys.stdout.write('Train/test split loaded successfully\n')
# reset index and get size of train/val 
X_train.reset_index(inplace = True, drop=True) 
y_train.reset_index(inplace = True, drop=True)
X_val.reset_index(inplace = True, drop=True)
y_val.reset_index(inplace = True, drop=True)
X_test.reset_index(inplace = True, drop=True)
y_test.reset_index(inplace = True, drop=True)
train_size = X_train.size
valid_size= X_val.size
test_size = X_test.size

# load images for train/val/test set
H = 64
W = 256
train_imgs = load_imgs(X_train, path_to_bb_segs, max_height=H, max_width=W)
val_imgs = load_imgs(X_val, path_to_bb_segs, max_height=H, max_width=W)
test_imgs = load_imgs(X_test, path_to_bb_segs, max_height=H, max_width=W)
train_imgs_rot = load_imgs(X_train, path_to_bb_segs, max_height=H, max_width=W, rotation=15)
val_imgs_rot = load_imgs(X_val, path_to_bb_segs, max_height=H, max_width=W, rotation=15)
test_imgs_rot = load_imgs(X_test, path_to_bb_segs, max_height=H, max_width=W, rotation=15)
sys.stdout.write('Image loading successfully')
sys.stdout.write('\n')
train_imgs = np.concatenate((train_imgs, train_imgs_rot), axis=0)
val_imgs = np.concatenate((val_imgs, val_imgs_rot), axis=0)
test_imgs = np.concatenate((test_imgs, test_imgs_rot), axis=0)
X_train = np.concatenate((X_train, X_train), axis=0)
X_val = np.concatenate((X_val, X_val), axis=0)
X_test = np.concatenate((X_test, X_test), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)
y_val = np.concatenate((y_val, y_val), axis=0)
y_test = np.concatenate((y_test, y_test), axis=0)
train_size = train_size * 2
valid_size= valid_size * 2
test_size = test_size * 2

# make alphabet
# here it includes all possible alphabets
# true means preset alphatbet + any additional labels appearing in dataset 
# false means no preset alphatbet (decrease alphabet size might improve model prediction accuracy)
alphabets = get_alphabets(y_train, True)

# define parameters for the length of input labels 
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels

# make train_y, train_label_len, train_input_len, train_output which are necessary for the CNN/RNN to understand the constraint of the model
train_y, train_label_len, train_input_len, train_output = label_helper(max_str_len, num_of_characters, num_of_timestamps, train_size, y_train, alphabets)
valid_y, valid_label_len, valid_input_len, valid_output = label_helper(max_str_len, num_of_characters, num_of_timestamps, valid_size, y_val, alphabets)
test_y, test_label_len, test_input_len, test_output = label_helper(max_str_len, num_of_characters, num_of_timestamps, test_size, y_test, alphabets)

sys.stdout.write('Total number of alphatbet {}'.format(len(alphabets)))
sys.stdout.write(alphabets)
sys.stdout.write('\n')



print()
####### Define Model #############
print(('####### Define Model #############'))

input_data = Input(shape=(H, W, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = Dropout(0.2)(inner)

inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max2')(inner)
inner = Dropout(0.2)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.2)(inner)

inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)
inner = Dropout(0.2)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max5')(inner)
inner = Dropout(0.2)(inner)


# CNN to RNN
inner = Reshape(target_shape=(H, W//(2**4)*128), name='reshape')(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

## RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm3')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm4')(inner)

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
lr = 0.001
epoch_size = int(args.numepochs)
batch_size = 64

final_report = final_report + 'Number of epochs: ' + args.numepochs + '\n' + 'Batch size: ' + str(batch_size) + '\n'

# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = lr))

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

model_final.fit(
    x=[train_imgs,train_y, train_input_len, train_label_len],
    y=train_output,
    shuffle=True,
    validation_data=([val_imgs, valid_y, valid_input_len, valid_label_len], valid_output),
    epochs=epoch_size,
    batch_size=batch_size,
    callbacks = [early_stopping])


# Make predictions for validation set and show results.
prediction = get_prediction(model,test_imgs,alphabets)
correct_info, final_report = get_prediction_accuracy(prediction, y_test, X_test, final_report)

if len(correct_info):
    final_report = final_report + 'List of correctly predicted words: \n'
    for i in correct_info:
        print('predicted: ', correct_info[i]['predicted'], '| label: ', correct_info[i]['label'])
        final_report = final_report + 'predicted: ' + correct_info[i]['predicted'] + '| label: ' + correct_info[i]['label'] +  '\n'

# Save model

cur_date = datetime.today().strftime('%Y-%m-%d')
model.save('models/' + cur_date + '_' + args.runtype + 'enriched_model22')
with open(f'run_summaries/{cur_date}_summaries.txt', 'a') as f:
    f.write(cur_date + '\n')
    f.write('Run type: ' + args.runtype.upper() + '\n'  )
    f.write(final_report + '----------------\n\n')
    f.close()
