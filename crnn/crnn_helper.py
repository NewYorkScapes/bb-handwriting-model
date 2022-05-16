import numpy as np
from collections import Counter
from tensorflow.keras import backend as K

def label_to_num(label,alphabets):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num,alphabets):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            continue
        else:
            ret+=alphabets[ch]
    return ret
    
    
# the ctc loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_alphabets(y_train, use_basic_alphabet = True):
  '''
  @params:
    y_train: labels for training data 
    y_val: labels for validation data
    use_basic_alphabet: Boolean. True if use basic a-z, A-Z, 1-9. False to only use characters appearing in training and validation sets 
  @return:
    alphabets
  '''
  # make alphabet
  # here it includes all possible alphabets
  all_string = ''
  for string in y_train:
    all_string += str(string)
    
  sort_chars = Counter(all_string).most_common()
  if use_basic_alphabet:
    alphabets = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-' "
  else:
    alphabets = "-' "

  for tupl in sort_chars:
    if tupl[0] not in alphabets:
      alphabets = tupl[0] + alphabets
  
  return alphabets
  

def label_helper(max_str_len, num_of_characters, num_of_timestamps, data_size, y_labels, alphabets):
  '''
  * **train_y** contains the true labels converted to numbers and padded with -1. The length of each label is equal to max_str_len. 
  * **train_label_len** contains the length of each true label (without padding) 
  * **train_input_len** contains the length of each predicted label. The length of all the predicted labels is constant i.e number of timestamps - 2.  
  * **train_output** is a dummy output for ctc loss. 
  
  @params: 
  max_str_len: max length of input labels
  num_of_characters: +1 for ctc pseudo blank
  num_of_timestamps: max length of predicted labels
  data_size: size of data
  y_labels: y_train or y_val
  
  @return
  train_y, train_label_len, train_input_len, train_output

  @Note: This applies not only to training set, but also validation set and test set. Don't get confused by the naming convetions. 
  '''
  train_y = np.ones([data_size, max_str_len]) * -1
  train_label_len = np.zeros([data_size, 1])
  train_input_len = np.ones([data_size, 1]) * (num_of_timestamps-2)
  train_output = np.zeros([data_size])

  for i in range(y_labels.size):
      train_label_len[i] = len(str(y_labels[i]))
      train_y[i, 0:len(str(y_labels[i]))]= label_to_num(str(y_labels[i]),alphabets)

  return train_y, train_label_len, train_input_len, train_output
  
 
 
def get_prediction(model,list_imgs,alphabets):
  '''
  @params:
  model: trained model for prediction 
  list_imgs: list of images to make prediction
  alphabets: alphabets defined before for all our training/validation sets
  @return:
  predictions: list of labels of predictions
  '''
  preds = model.predict(list_imgs)
  decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])
  prediction = []
  for i in range(len(list_imgs)):
      prediction.append(num_to_label(decoded[i],alphabets))

  return prediction

def cer(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    cer = distances[-1] / len(s2)
    return cer

def get_prediction_accuracy(prediction, y_true, X_val, final_report):
  '''
  @params:
  prediction: prediction made by model 
  y_true: labels for the predictions
  X_val: index for validation set  
  @return:
  correct_info: dict of segments where at least one character is predicted correctly. 
  '''

  correct_char = 0
  total_char = 0
  correct = 0
  edit_distance_total = 0
  corr_char_segid = []
  correct_info = {}
  for i in range(len(prediction)):
      pr = prediction[i]
      tr = y_true[i]
      edit_distance = cer(pr, tr) * 100
      edit_distance_total += edit_distance
      print(pr, '|', tr, edit_distance)
      try:
        total_char += len(tr)
        # Some characters are somehow turning up as floats (numbers? the fractions?) so they don't have a length.
        # I punt here and add a single character to length to total count just to keep numbers accurate.
        # Since the character is unusual (a float or non-alphanumeric character) it should be skipped anyway as an incorrect label
      except:
        total_char +=1
      try:
          for j in range(min(len(tr), len(pr))):
              if tr[j] == pr[j]:
                  correct_char += 1
                  if i not in corr_char_segid:
                    corr_char_segid += [i]
      except:
          print("An unusual character was found in the 'true value': ", tr)
      if pr == tr :
          correct += 1
          # record any correct predicted full word
          correct_info[X_val[i]] = {'predicted': pr, 'label': tr}

  print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
  print('Correct words predicted      : %.2f%%' %(correct*100/len(X_val)))
  print('Average edit distance      : %.2f%%' % (edit_distance_total/ len(X_val)))
  final_report += 'Correct characters predicted : %.2f%%' %(correct_char*100/total_char) + '\n'
  final_report += 'Correct words predicted      : %.2f%%' %(correct*100/len(X_val)) + '\n'
  final_report += 'Average edit distance     : %.2f%%' % (edit_distance_total/ len(X_val)) + '\n'
  
  return correct_info, final_report
