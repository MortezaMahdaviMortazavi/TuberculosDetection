import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_filepaths_and_labels(sdir):
    filepaths = []
    labels = []
    classlist = sorted(os.listdir(sdir))
    for _class in classlist:
        classpath = os.path.join(sdir, _class)
        if os.path.isdir(classpath):
            flist = sorted(os.listdir(classpath))
            for f in tqdm(flist, ncols=130, desc=f'{_class:25s}', unit='files', colour='blue'):
                fpath = os.path.join(classpath, f)
                filepaths.append(fpath)
                labels.append(_class)
    return filepaths, labels

def create_dataframes(filepaths, labels):
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df

def split_data(df):
    train_df, dummy_df = train_test_split(df, train_size=.8, shuffle=True, random_state=123, stratify=df['labels'])
    valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])
    return train_df, test_df, valid_df

def calculate_average_image_size(df, num_samples=50):
    sample_df = df.sample(n=num_samples, replace=False)
    ht = 0
    wt = 0
    count = 0
    for i in range(len(sample_df)):
        fpath = sample_df['filepaths'].iloc[i]
        try:
            img = cv2.imread(fpath)
            h, w, _ = img.shape
            wt += w
            ht += h
            count += 1
        except:
            pass
    average_height = int(ht / count)
    average_weight = int(wt / count)
    aspect_ratio = average_height / average_weight
    return average_height,average_weight,aspect_ratio


def make_dataframes(sdir):
    filepaths, labels = get_filepaths_and_labels(sdir)
    df = create_dataframes(filepaths, labels)
    train_df, test_df, valid_df = split_data(df)
    average_height, average_weight, aspect_ratio = calculate_average_image_size(train_df)
    
    # Other statistics and information can be printed here if needed.
    class_count = len(train_df['labels'].unique())
    counts = list(train_df['labels'].value_counts())
    
    return train_df, test_df, valid_df, class_count, average_height, average_weight, aspect_ratio

