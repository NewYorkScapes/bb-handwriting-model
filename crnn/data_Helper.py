import pandas as pd
import numpy as np
import os
import sys
import urllib.request
from collections import Counter
from sklearn.model_selection import train_test_split
import cv2
import argparse
from scipy import ndimage
from PIL import Image, ImageFont, ImageDraw


def parse_argument():
  parser = argparse.ArgumentParser(prog='python main.py',
                                   usage='%(prog)s [-h] [-r <run-type>]',
                                   description='A CRNN OCR script'
                                   )
  parser.add_argument('-r', '--runtype', action='store', required=True,
                      help="""Type of run desired (Must be one of 'validated', 'validated_onepass', 'validated_iam', 'validated_onepass_iam' , 'onepass_iam')"""
                      )

  parser.add_argument('-f', '--trainedfilename', action='store', required=True,
                      help="""File name of trained data downloaded from transcriber"""
                      )

  parser.add_argument('-e', '--numepochs', action='store', required=True, help="""Number of epochs to be deployed in run""")

  return parser.parse_args()


def upload_segs(path, path_to_save_segs):
  '''
  @params:
    path: path to the csv file
    path_to_save_segs: path to save all the segments
    the csv must contains 'transcription', 'segment_id' and 'seg_url' columns for naming convention
  @ return:
    boolean: True if all segments are uploaded successfully
  '''

  transcribe_df  = pd.read_csv(path)
  # drop nan 
  transcribe_df.dropna(inplace=True,subset=['transcription'])
  sys.stdout.write('Total # of segments in this csv: {}'.format( len(transcribe_df)))
  sys.stdout.write('\n')

  id_url = transcribe_df[['segment_id','seg_url']]
  for row_num in range(len(id_url)):
    jpg_list = os.listdir(path_to_save_segs)
    if row_num % 500  == 0:
      sys.stdout.write('{} out of {} Done'.format(row_num, len(id_url)))
      sys.stdout.write('\n')
    try:
      url = id_url.loc[row_num].seg_url
      id = id_url.loc[row_num].segment_id
      if '{}.jpg'.format(id) in jpg_list:
        # if seg already exists in folder, continue
        continue
      urllib.request.urlretrieve(url, '{}/{}.jpg'.format(path_to_save_segs,id))
    except:
      pass
  return True



  # data loading, it should handle 1. only validated data 2. validated data + one pass data 3. iamdata set

def load_data(data_type_used,path_to_seg_csv,path_to_bb_segs,path_to_iam_csv = None):
  '''
  @Params:
    data_type_used = Must be one of 'validated', 'validated_onepass', 'validated_iam', 'validated_onepass_iam' , 'onepass_iam'
    path_to_seg_csv = path to csv where it contains all info of our transcribed segs.
    path_to_bb_segs = path to all transcribed segments we uploaded, identified by its unique segment_id. For example, if segment_id for a segment is 12315, then segment name is 12315.jpg
    path_to_iam_csv = path to csv where it contais all iam images we want to use
  @Return:
    Pandas.Dataframe():
      segment_id: the segment id for each segment
      label: the corresponding transcribed label for each segment
  '''

  if data_type_used not in ['validated', 'validated_onepass', 'validated_iam', 'validated_onepass_iam' , 'onepass_iam']:
    raise Exception('Please use one of the following: validated, validated_one_pass, validated_iam, validated_one_pass_iam , one_pass_iam' )
  
  # LOAD CSV CONTAINING INFO OF TRANSCRIBED DATA
  transcribe_df  = pd.read_csv(path_to_seg_csv)
  # drop nan 
  transcribe_df.dropna(inplace=True,subset=['transcription'])

  # boolean to check if we already have a dataframe
  df_exist = False
  
  if 'validated' in data_type_used:
    df_exist = True
    df = pd.DataFrame()
    # groupby segment_id and check every segment with > 1 transcription and there is no discrepancy among transcription. 
    # Note that if transcription = 3 and two transcriptions are the same, this does not count for discrepancy. Same apply to all other segments. 
    transcription_by_id = transcribe_df.groupby('filename')['transcription'].apply(list)
    seg_id = []
    seg_val = []
    for idx in transcription_by_id.index:
      if len(transcription_by_id[idx]) > 1:
        if pd.Series(transcription_by_id[idx]).unique().size != len(transcription_by_id[idx]): 
          seg_id.append(idx)
          seg_val.append(Counter(transcription_by_id[idx]).most_common()[0][0])
    sys.stdout.write('We have {} validated data'.format(len(seg_id)))
    sys.stdout.write('\n')

    df['filename'] = seg_id
    df['label'] = seg_val

  if 'onepass' in data_type_used:
    transcription_by_id = transcribe_df.groupby('filename')['transcription'].apply(list)
    # select one pass segs
    one_pass_idlist = []
    for idx in transcription_by_id.index:
      if len(transcription_by_id[idx]) == 1:
          one_pass_idlist += [idx]

    one_pass_df = transcribe_df[transcribe_df['filename'].isin(one_pass_idlist )]
    one_pass_df = one_pass_df[['filename', 'transcription']]
    one_pass_df.columns = ['filename','label']
    
    sys.stdout.write('We have {} one pass data'.format(len(one_pass_df)))
    sys.stdout.write('\n')
    if df_exist == True:
      df = pd.concat([df,one_pass_df])
      df = df[['filename', 'label']]
    else:
      df = one_pass_df

  # need to delete rows where image files are not in folder 
  all_img_files = os.listdir(path_to_bb_segs)
  segs_not_in_folder  = 0
  for seg_id in df['filename'].unique():
    if seg_id not in all_img_files:
      segs_not_in_folder += 1
      df = df[df['filename'] != seg_id]
  sys.stdout.write('Segs not in folder: {}'.format(segs_not_in_folder))
  sys.stdout.write('\n')

  if 'iam' in data_type_used:
    iam_df = pd.read_csv(path_to_iam_csv)
    iam_df.columns = ['	Unnamed: 0','segment_id','label']
    # pick a subset of iam_df with len(iam) = len(validated_df)
    iam_df = iam_df.iloc[:len(df)]
    iam_df = iam_df[['segment_id','label']]
    iam_df.columns = ['filename', 'label']
    sys.stdout.write('We have {} iam data'.format(len(iam_df)))
    sys.stdout.write('\n')
    df = pd.concat([df,iam_df])

  # NO repeated segment_id is allowed 
  assert len(df['filename'].unique()) == len(df)

  sys.stdout.write('Total # of data in our df: {}'.format(len(df)))
  sys.stdout.write('\n')

  return df



def data_train_val_split(df, IAM_USED = False, test_size = 0.1, random_state = 42 ):
    '''
    @Params:
     IAM_USED: Boolean: if iam data used, IAM_USED = True
     df: pd.DataFrame() where it contains all segment_id and label
     test_size: float, fraction of data used for validation set
     random_state: default = 42, used to track the ramdom split data.

    @Return:
      X_train, X_val, y_train, y_val
    '''

    # split train/val/test

    X_train, X_val, y_train, y_val  = train_test_split(
    df['filename'], df['label'], test_size=test_size, random_state= random_state)

    sys.stdout.write('training sample size: {}'.format(X_train.size))
    sys.stdout.write('\n')
    sys.stdout.write('Vallidation sample size: {}'.format(X_val.size))
    sys.stdout.write('\n')

    return X_train, X_val, y_train, y_val


def get_stats_discrepancy(path):
  '''
  @params:
    path: path of transcription csv file 

  @return:
    dict: a dictionary where key is the segment_id and value is the discrepancy of segments 
  '''
  transcribe_df  = pd.read_csv(path)
  # drop nan 
  transcribe_df.dropna(inplace=True,subset=['transcription'])

  # multiple passes stats
  stats = dict(transcribe_df.groupby('filename').count()['seg_url'])
  stats = pd.Series(stats.values(),index = stats.keys())
  # number of passes
  print('We have this unique number of passes: ', stats.unique() )
  unique_pass = stats.unique()
  for count in unique_pass:
    sys.stdout.write('For {} pass we have {} number of segments'.format(count, stats[stats == count].size ))
    print()
  
  # discrepancy
  transcription_by_id = transcribe_df.groupby('filename')['transcription'].apply(list)
  # check for discrepancy
  discrepancy_idlist = []
  for idx in transcription_by_id.index:
    if len(transcription_by_id[idx]) > 1:
      if pd.Series(transcription_by_id[idx]).unique().size == len(transcription_by_id[idx]): # this statement only includes transcription that does not have duplicate values
      # For example, if a segment is translated 3 times, among them two values are the same, so we will treat the two values as duplicate(correct) value and not include this segment into discrepancy list
        discrepancy_idlist += [idx]
        # print('Transcription of idx {} has discrepancy {}'.format(idx, transcription_by_id[idx]))
  discrepancy_dict = {}
  for i,idx in enumerate(discrepancy_idlist):
      discrepancy_dict[idx] = transcription_by_id[idx]
  sys.stdout.write('We have {} number of segments that have discrepancy'.format(len(discrepancy_dict)))
  print()
  
  return discrepancy_dict



def preprocess(img, max_width = 256, max_height = 64, turn_grey = True, rotation=0, thres='default'):
    '''
    @param
    img:cv2 image object 
    max_width = max_width of img
    max_height = max_height of img

    @return 
    a cropped/enlarged segment with size 
    '''
    # convert to balck and white
    if turn_grey:
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, sharpen_kernel)
        if thres=='binary':
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
        else:
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]
    # rotate by 15 degree anticlockwise
    img = ndimage.rotate(img, rotation, reshape = True, cval=255)

    (h, w) = img.shape
    
    final_img = np.ones([max_height, max_width])*255 # blank white image

    if h > max_height:
        scale_factor = max_height/h
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        (h, w) = img.shape

    if w > max_width:
        w = max_width
        img = img[:, :w]

    final_img[:h, :w] = img

    # normalize matrix value
    final_img = final_img / 255.

    return final_img


def load_imgs(img_id_list, path_to_bb_segs, max_width = 256, max_height = 64, rotation=0, thres='default'):
  '''
  @params
  img_id_list: X_train or X_val or X_test for our data
  path_to_bb_segs: path to browns brother trasncribed segments
  path_to_iam_segs: path to iam segments
  max_width: max_width of cropped img, default 256
  max_height: max_height of cropped img, default 64
  
  Note: the naming for iam can be whatever you like, but please include non-digit character in the naming, 
  so that the code can differentiate where to find the correct location of images if iam data included. 

  @return
  list: list of images(matrix) for model to train/val/test
  '''
  processed_imgs = []
  for i,idx in enumerate(img_id_list):
      if i % 500 == 0:
        sys.stdout.write('{} out of {} done \n'.format(i,len(img_id_list)))
      img_dir = '{}/{}'.format(path_to_bb_segs, idx)
      image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
      if image is None:
        raise Exception('None in dataset')
      image = preprocess(image ,max_width = max_width, max_height = max_height, rotation = rotation )
      processed_imgs.append(image)
  return processed_imgs


def generate_image(text, font):
  image = Image.new('L', (256, 72), color=256)
  drawing = ImageDraw.Draw(image)
  drawing.text((0, 0), text, fill=0, font=font)
  return image

