import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import mean_squared_error
from math import sqrt
import h5py
import json
__name__ = "__main__" 
def load_train() -> pd.DataFrame:
    return pd.read_csv(os.path.join('data', 'train.csv'))

def load_pedigree() -> pd.DataFrame:
    return pd.read_csv(os.path.join('data', 'pedigree.csv'))

def concate_train_and_pedigree(train, pedigree_df) -> pd.DataFrame:
    train= train.merge(pedigree_df, how='left', on='animal_id')
    return train

def Nan_to_avg_milk_yeld(train) -> pd.DataFrame:
    for row_index, row_with_nan in train[train.isnull().any(1)].iterrows():
        row_with_nan = row_with_nan.fillna(0)[6:16]

        averange_milk_yeld=row_with_nan.sum()/(10- list(row_with_nan).count(0))
        row_with_nan = row_with_nan.replace(0,averange_milk_yeld)

        train.loc[row_index,'milk_yield_1':'milk_yield_10'] = row_with_nan
    return train

def add_mother_milk_yeld(daughter_df, mother_df, train) -> pd.DataFrame:
  count=0
  for mother_id in daughter_df['mother_id']:
    if count%10000==0:
      print(count)
    count+=1
    for animal_id in mother_df['animal_id']:
      if animal_id==mother_id:
        indexes = train.index[train['animal_id']==animal_id]
        # print('Индексы строк матери: {}'.format(indexes))
        indexes_daughter = train.index[train['mother_id']==mother_id]
        # print('Индексы строк дочерей: {}'.format(indexes_daughter))
        if len(indexes)>1:
          averange_mother_milk_by_lactation_list=[]
          for variable in range(6,16):
            averange_milk_yeld_by_lactation=0
            for x in range(len(indexes)):
              averange_milk_yeld_by_lactation+= train.loc[indexes[x]][variable]

            averange_milk_yeld_by_lactation=averange_milk_yeld_by_lactation/len(indexes)
            averange_mother_milk_by_lactation_list.append(averange_milk_yeld_by_lactation)

          train.loc[indexes_daughter,'mother_milk_yield_1':'mother_milk_yield_10']=averange_mother_milk_by_lactation_list
          break
        else:
          train.loc[indexes_daughter,'mother_milk_yield_1':'mother_milk_yield_10']=train.loc[indexes]
          break
  return train

def family_tree(train,pedigree_df) -> pd.DataFrame:
    train = train.reindex(train.columns.tolist() + ['mother_milk_yield_1', 'mother_milk_yield_2', 'mother_milk_yield_3','mother_milk_yield_4', 'mother_milk_yield_5','mother_milk_yield_6', 'mother_milk_yield_7','mother_milk_yield_8', 'mother_milk_yield_9', 'mother_milk_yield_10'], axis=1)
    pedigree_2=pd.DataFrame(pedigree_df['mother_id'])
    pedigree_2 = pedigree_2.rename(columns={'mother_id': 'animal_id'})
    train_1_generation=pd.merge(train, pedigree_2, on=['animal_id'], how='inner')
    train_1_generation = train_1_generation.drop_duplicates()
    print('train_1_generation.shape = {}'.format(train_1_generation.shape))

    train_2_generation = pd.DataFrame(train_1_generation['animal_id'])
    train_2_generation = train_2_generation.rename(columns={'animal_id': 'mother_id'})
    train_2_generation = pd.merge(train, train_2_generation, on=['mother_id'], how='inner')
    train_2_generation = train_2_generation.drop_duplicates()
    print('train_2_generation.shape = {}'.format(train_2_generation.shape))
    train = add_mother_milk_yeld(train_2_generation, train_1_generation, train)

    train_3_generation= pd.DataFrame(train_2_generation['animal_id'])
    train_3_generation = train_3_generation.rename(columns={'animal_id': 'mother_id'})
    train_3_generation = pd.merge(train, train_3_generation, on=['mother_id'], how='inner')
    train_3_generation = train_3_generation.drop_duplicates()
    print('train_3_generation.shape = {}'.format(train_3_generation.shape))
    train = add_mother_milk_yeld(train_3_generation, train_2_generation, train)

    train_4_generation= pd.DataFrame(train_3_generation['animal_id'])
    train_4_generation = train_4_generation.rename(columns={'animal_id': 'mother_id'})
    train_4_generation = pd.merge(train, train_4_generation, on=['mother_id'], how='inner')
    train_4_generation = train_4_generation.drop_duplicates()
    print('train_4_generation.shape = {}'.format(train_4_generation.shape))
    train = add_mother_milk_yeld(train_4_generation, train_3_generation, train)

    train_5_generation= pd.DataFrame(train_4_generation['animal_id'])   
    train_5_generation = train_5_generation.rename(columns={'animal_id': 'mother_id'})
    train_5_generation = pd.merge(train, train_5_generation, on=['mother_id'], how='inner')
    train_5_generation = train_5_generation.drop_duplicates()
    print('train_5_generation.shape = {}'.format(train_5_generation.shape))
    train = add_mother_milk_yeld(train_5_generation, train_4_generation, train)
    
    return train
    #return mega_train

def  load_mega_train()->pd.DataFrame:
    return pd.read_csv(os.path.join('data', 'mega_train.csv'))


def creating_train_mothers_and_sisters(train)-> pd.DataFrame:
    train[['mother_id','father_id']]=train[['mother_id','father_id']].fillna(0)
    train_mothers_and_sisters = train.reindex(train.columns.tolist() + ['sister_milk_yield_1', 'sister_milk_yield_2', 'sister_milk_yield_3','sister_milk_yield_4', 'sister_milk_yield_5','sister_milk_yield_6', 'sister_milk_yield_7','sister_milk_yield_8', 'sister_milk_yield_9', 'sister_milk_yield_10'], axis=1)
    return train_mothers_and_sisters

def creating_df_sisters(train_mothers_and_sisters)-> pd.DataFrame:
    cols=['mother_id','father_id']
    train_sisters = train_mothers_and_sisters[train_mothers_and_sisters.duplicated(cols, keep=False) == True]
    train_sisters=train_sisters.drop_duplicates (subset=['animal_id'])
    return train_sisters

def add_sisters_milk_yeld(train_sisters, train, train_mothers_and_sisters)-> pd.DataFrame:
    #Нахождение и добавление сестер в стоку, предварительно создав для этого столбцы
    cols=['mother_id','father_id']
    parents=train_sisters[cols].iloc[0]
    columns=train_sisters.columns[6:16]
    avg_sisters_milk_lactation=np.zeros(10, dtype = float)
    count_sisters=0
    count=0
    for index, row in train_sisters.iterrows():
        count+=1
        if count%10000==0:
            print(count)
        if (parents.equals(row[cols])) == False:
            avg_sisters_milk_lactation/=count_sisters
            count_sisters=1

            indexes_mother = train.index[train['mother_id']==parents[0]]
            indexes_father = train.index[train['father_id']==parents[1]]
            indexes=list(set(indexes_mother) & set(indexes_father))

            parents=row[cols]
            train_mothers_and_sisters.loc[indexes,'sister_milk_yield_1':'sister_milk_yield_10']=avg_sisters_milk_lactation

            avg_sisters_milk_lactation=np.zeros(10, dtype = float)
            avg_sisters_milk_lactation= row[columns].to_numpy()

        else:
            count_sisters+=1
            for x in range(10):
                avg_sisters_milk_lactation[x]+= row[6+x]
    
    return train_mothers_and_sisters
    #return mega_sisters_train

def load__mega_sisters_train() -> pd.DataFrame:
    return pd.read_csv(os.path.join('data', 'mega_sisters_train.csv'))


def date_to_normal(date_column, cols, train_mothers_and_sisters)-> pd.DataFrame:
  for index, date in enumerate(date_column):
    year_month_day = [time for time in date.split('-')]
    for i, time  in enumerate(year_month_day):
      if time[0]=='0':
        time= time[1:]
      year_month_day[i]= time
    train_mothers_and_sisters.loc[index, cols] = year_month_day
  return train_mothers_and_sisters

def date_in_columns(train_mothers_and_sisters)-> pd.DataFrame:
    calving_date = train_mothers_and_sisters['calving_date']
    cols=['calving_year','calving_month', 'calving_day']
    train_mothers_and_sisters = train_mothers_and_sisters.reindex(train_mothers_and_sisters.columns.tolist() + cols, axis=1)
    train_mothers_and_sisters = date_to_normal(calving_date, cols, train_mothers_and_sisters)
    train_mothers_and_sisters.drop(columns=['calving_date'], inplace=True)

    birth_date = train_mothers_and_sisters['birth_date']
    cols=['birth_year','birth_month', 'birth_day']
    train_mothers_and_sisters = train_mothers_and_sisters.reindex(train_mothers_and_sisters.columns.tolist() + cols, axis=1)
    train_mothers_and_sisters = date_to_normal(birth_date, cols, train_mothers_and_sisters)
    train_mothers_and_sisters.drop(columns=['birth_date'], inplace=True)
    return  train_mothers_and_sisters


def change_number_of_decimal_places(train_mothers_and_sisters)-> pd.DataFrame:
    #Уменьшаем кол-во знаков после запятой до двух
    train_mothers_and_sisters[train_mothers_and_sisters.columns[4:14]] = train_mothers_and_sisters[train_mothers_and_sisters.columns[4:14]].apply(lambda x: round(x, 3))
    train_mothers_and_sisters[train_mothers_and_sisters.columns[16:36]] = train_mothers_and_sisters[train_mothers_and_sisters.columns[16:36]].apply(lambda x: round(x, 3))
    return train_mothers_and_sisters
#return final_dataset_train
# проконтролировать чтобы final_dataset = train_mothers_and_sisters

def load_final_dataset_train() -> pd.DataFrame:
    final_dataset= pd.read_csv(os.path.join('data', 'final_dataset_train.csv'))
    final_dataset=final_dataset.fillna(0)
    return final_dataset

def load_test_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)

def concate_test_and_pedigree(test_df)-> pd.DataFrame:
    final_dataset=  load_final_dataset_train()
    pedigree_df = load_pedigree()

    test= test_df.merge(pedigree_df, how='left', on='animal_id')
    if 'Unnamed: 0' in test.columns:
        test.drop(columns='Unnamed: 0', inplace=True)
    test.drop(columns='father_id', inplace=True)

    final_dataset_for_test = final_dataset.drop(columns = ['animal_id', 'lactation', 'farm', 'farmgroup', 'milk_yield_1',
       'milk_yield_2', 'milk_yield_3', 'milk_yield_4', 'milk_yield_5',
       'milk_yield_6', 'milk_yield_7', 'milk_yield_8', 'milk_yield_9',
       'milk_yield_10', 'father_id','calving_year',	'calving_month','calving_day','birth_year','birth_month','birth_day'])
    final_test = pd.concat([test,final_dataset_for_test], axis=1, join='outer')
    final_test = final_test.dropna (subset=['animal_id'])
    final_test= final_test.drop(columns=['mother_id', 'animal_id'])

    return final_test

def change_date(final_test)-> pd.DataFrame:
    calving_date=final_test['calving_date']
    cols=['calving_year','calving_month', 'calving_day']
    final_test = final_test.reindex(final_test.columns.tolist() + cols, axis=1)
    final_test = date_to_normal(calving_date, cols, final_test)
    final_test.drop(columns=['calving_date'], inplace=True)

    birth_date = final_test['birth_date']
    cols=['birth_year','birth_month', 'birth_day']
    final_test= final_test.reindex(final_test.columns.tolist() + cols, axis=1)
    final_test= date_to_normal(birth_date, cols, final_test)
    final_test.drop(columns=['birth_date'], inplace=True)
    return final_test

def my_train_test_split(final_dataset):
    # Разбиваем df на Train and Test
    Y=final_dataset[final_dataset.columns[6:14]]
    X=final_dataset.drop(columns =final_dataset.columns[6:14])

    X_train, X_val, y_train, y_val = train_test_split(X,Y, random_state=42, test_size=0.25)
    return X_train, X_val, y_train, y_val

def drop_bad_columns(X_train, X_val):
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    good_label_cols = [col for col in object_cols if set(X_val[col]).issubset(set(X_train[col]))]
    bad_label_cols = list(set(object_cols)-set(good_label_cols))
    X_train.drop(columns=bad_label_cols, inplace=True)
    X_val.drop(columns=bad_label_cols, inplace=True)
    return X_train, X_val

def normalization_data(X_train, X_val, final_test):
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # Columns that will be one-hot encoded
    low_cardinality_cols = [col for col in X_train.columns if X_train[col].nunique() < 10]

    df_low_cardinality_cols_train    = X_train[low_cardinality_cols]
    df_low_cardinality_cols_train_OH = pd.DataFrame(OH_encoder.fit_transform(df_low_cardinality_cols_train))
    df_low_cardinality_cols_val      = X_val[low_cardinality_cols]
    df_low_cardinality_cols_val_OH   = pd.DataFrame(OH_encoder.transform(df_low_cardinality_cols_val))

    final_test_lactation= final_test[low_cardinality_cols]
    final_test_OH   = pd.DataFrame(OH_encoder.transform(final_test_lactation))
# One-hot encoding removed index
    df_low_cardinality_cols_train_OH.index =  X_train.index
    df_low_cardinality_cols_val_OH.index   =  X_val.index
    final_test_OH.index   =  final_test.index
# Remove columns
    X_train= X_train.drop(columns=low_cardinality_cols)
    X_val= X_val.drop(columns=low_cardinality_cols)
    final_test= final_test.drop(columns=low_cardinality_cols)
# Add one-hot encoded columns to numerical features
    X_train=pd.concat([X_train, df_low_cardinality_cols_train_OH],axis=1)
    X_val=pd.concat([X_val, df_low_cardinality_cols_val_OH], axis=1)
    final_test=pd.concat([final_test, final_test_OH], axis=1)

    X_train.columns=X_train.columns.astype(str)
    X_val.columns=X_val.columns.astype(str)
    final_test.columns=final_test.columns.astype(str)
    return X_train, X_val, final_test
    
def my_fit(X_train, y_train, X_val, y_val):
    model = keras.Sequential([
        layers.BatchNormalization(input_shape = [X_train.shape[1]]),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(8),

        ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.MeanSquaredError()
        )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=7,
        min_delta=0.001
        )
    
    model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=256,
    epochs=7,
    callbacks=[early_stopping],
    verbose=1,
    )
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights("model.h5")

    return  model

def my_predict(model, dataset, path) -> pd.DataFrame:
    
    unmodified_test = pd.read_csv(path)
    dataset = dataset.astype(np.float32)
    predictions = model.predict(dataset)

    X_COLUMNS_TEST = ['milk_yield_3','milk_yield_4', 'milk_yield_5', 'milk_yield_6', 'milk_yield_7', 'milk_yield_8', 'milk_yield_9', 'milk_yield_10']
    unmodified_test= unmodified_test.drop(columns=['farm', 'farmgroup','milk_yield_1','milk_yield_2', 'calving_date','birth_date'])
    
    # Преобразовать выдачу модели в читаемый прогноз
    x_answer = pd.DataFrame(predictions, columns=X_COLUMNS_TEST)
    submission=pd.concat([unmodified_test, x_answer], axis=1, ignore_index=True)
    submission.drop(columns=[0,1,2], inplace=True)
    submission = submission.rename({3:'milk_yield_3', 4:'milk_yield_4', 5:'milk_yield_5', 6: 'milk_yield_6',7:'milk_yield_7', 8:'milk_yield_8', 9:'milk_yield_9', 10:'milk_yield_10'}, axis=1)
    return submission


if __name__ == '__main__':
#    train = load_train()
#    pedigree = load_pedigree()
#    train = concate_train_and_pedigree(train, pedigree)
#    train = Nan_to_avg_milk_yeld(train)
#    train = family_tree(train,pedigree)
#
#    mega_train =load_mega_train()
#    train_mothers_and_sisters = creating_train_mothers_and_sisters(train)
#    train_sisters= creating_df_sisters(train_mothers_and_sisters)
#    train_mothers_and_sisters = add_sisters_milk_yeld(train_sisters, train, train_mothers_and_sisters)
#
#    train_mothers_and_sisters = load__mega_sisters_train() 
#    train_mothers_and_sisters = date_in_columns(train_mothers_and_sisters)
#
#    final_dataset = change_number_of_decimal_places(train_mothers_and_sisters)
# Сверху представлены функции по прдобработки первоначального train, т.к их выполнение занимает много времени, я уже загружаю предпоготовленный dataset
    final_dataset = load_final_dataset_train()

    X_train, X_val, y_train, y_val = my_train_test_split(final_dataset)
    X_train, X_val = drop_bad_columns(X_train, X_val)

    test_df = load_test_dataset(os.path.join('data', 'X_test_public.csv'))
    final_test= concate_test_and_pedigree(test_df)
    final_test = change_date(final_test)
    X_train, X_val, final_test = normalization_data(X_train, X_val, final_test)


    model = my_fit(X_train, y_train, X_val, y_val)

    
    _submission = my_predict(model, final_test, os.path.join('data', 'X_test_public.csv'))
    _submission.to_csv(os.path.join('data', 'submission.csv'), sep=',', index=False)


    # X_train, X_val, y_train, y_val = my_train_test_split(final_dataset)
    # X_train, X_val = drop_bad_columns(X_train, X_val)

    # test_df = load_test_dataset(os.path.join('..', 'external', 'private', 'X_test_private.csv'))
    # final_test= concate_test_and_pedigree(test_df)
    # final_test = change_date(final_test)
    # X_train, X_val, final_test = normalization_data(X_train, X_val, final_test)

    # _submission_private = my_predict(model, final_test, os.path.join('..', 'external', 'private', 'X_test_private.csv'))
    # _submission_private.to_csv(os.path.join('data', 'submission_private.csv'), sep=',', index=False)
