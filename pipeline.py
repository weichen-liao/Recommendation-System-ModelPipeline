# -*- coding: utf-8 -*-
# Author: Weichen Liao

from surprise import Dataset
from surprise.dataset import DatasetAutoFolds
from pathlib import Path
from surprise import Reader
import pandas as pd
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.prediction_algorithms.knns import KNNBasic
from surprise import SVD
from surprise.trainset import Trainset
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise import accuracy
from surprise.model_selection import train_test_split
import time


# options for algo: 'SVD', 'KNN', 'NMF'
def get_trained_model(algo: str, model_kwargs: dict, train_set: Trainset) -> AlgoBase:
    if algo not in ['SVD', 'KNN', 'NMF']:
        raise Exception('algo only support: SVD, KNN, NMF')
    if algo == 'KNN':
        model = KNNBasic(k=model_kwargs['k'], min_k=model_kwargs['min_k'], sim_options = model_kwargs['sim_options'])
    elif algo == 'NMF':
        model = NMF(n_factors=model_kwargs['n_factors'], n_epochs=model_kwargs['n_epochs'], verbose=model_kwargs['verbose'])
    else:
        model = SVD()
    time_start = time.time()
    model.fit(train_set)
    time_end = time.time()
    cost_time = (time_end-time_start)*1000
    return model, cost_time

def load_ratings_from_surprise() -> DatasetAutoFolds:
    ratings = Dataset.load_builtin('ml-100k')
    return ratings

def load_ratings_from_file(ratings_filepath : Path) -> DatasetAutoFolds:
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    ratings = Dataset.load_from_file(ratings_filepath, reader)
    return ratings

def get_data(from_surprise : bool = True, file_path:str = '../data/movielens/ml-latest-small/ratings.csv') -> DatasetAutoFolds:
    if from_surprise:
        data = load_ratings_from_surprise() if from_surprise else load_ratings_from_file()
    else:
        ratings_filepath = Path(file_path)
        data = load_ratings_from_file(ratings_filepath)
    return data

def evaluate_model(model: AlgoBase, test_set: [(int, int, float)]) -> dict:
    predictions = model.test(test_set)
    metrics_dict = {}
    metrics_dict['RMSE'] = accuracy.rmse(predictions, verbose=False)
    metrics_dict['MAE'] = accuracy.rmse(predictions, verbose=False)
    return metrics_dict

def train_and_evalute_model_pipeline(algo: str, model_kwargs: dict = {},
                                     from_surprise: bool = True,
                                     test_size: float = 0.2) -> (AlgoBase, dict):
    data = get_data(from_surprise)
    train_set, test_set = train_test_split(data, test_size, random_state=42)
    model, cost_time = get_trained_model(algo, model_kwargs, train_set)
    metrics_dict = evaluate_model(model, test_set)
    return model, metrics_dict, cost_time


# def benchmark(name: str, algo: str, model_kwargs: dict = {}, from_surprise: bool = True, test_size: float = 0.2):
'''
example of model_dict:
{'KNN user based cosine': 
    {'algo': 'KNN', 'model_kwargs':
        {'k':40, 'min_k':1 ,'sim_options': {'user_based': False, 'name': 'cosine'}},
        'from_surprise':True, 'test_size':0.2},
 'KNN user based pearson':
    {'algo': 'KNN', 'model_kwargs':
        {'k':40, 'min_k':1 ,'sim_options': {'user_based': False, 'name': 'pearson'}},
        'from_surprise':True, 'test_size':0.2},
 'NMF user based':
    {'algo': 'NMF', 'model_kwargs':
        {'n_factors':15, 'n_epochs': 50, 'verbose':False},
        'from_surprise':True, 'test_size':0.2},
 'SVD user based':
    {'algo': 'SVD', 'model_kwargs':{}, 'from_surprise':True, 'test_size':0.2}
}
'''
def benchmark(try_dict: dict) -> (pd.DataFrame, dict):
    arr_res = []
    model_dict = {}
    for key in try_dict.keys():
        print('------------------processing',key,'------------------')
        item = try_dict[key]
        model, metrics_dict, cost_time = train_and_evalute_model_pipeline(algo=item['algo'], model_kwargs=item['model_kwargs'], from_surprise=item['from_surprise'], test_size=item['test_size'])
        RMSE, MAE = metrics_dict['RMSE'], metrics_dict['MAE']
        arr_res.append([key, RMSE, MAE, cost_time])
        model_dict[key] = model
    df_res = pd.DataFrame(arr_res, columns=['model name', 'RMSE', 'MAE', 'fit_time(ms)'])
    return df_res, model_dict

def get_user_recommendation(model: AlgoBase, user_id: int, k: int, data, movies: pd.DataFrame) -> pd.DataFrame:
    """Makes movie recommendations a user.

    Parameters
    ----------
        model : AlgoBase
            A trained surprise model
        user_id : int
            The user for whom the recommendation will be done.
        k : int
            The number of items to recommend.
        data : FIXME
            The data needed to do the recommendation.
        movies : pandas.DataFrame
            The dataframe containing the movies metadata (title, genre, etc)

    Returns
    -------
    pandas.Dataframe
        A dataframe with the k movies that will be recommended the user. The dataframe should have the following
        columns (movie_name : str, movie_genre : str, predicted_rating : float, true_rating : float)

    Notes
    -----
    - You should create other functions that are used in this one and not put all the code in the same function.
        For example to create the final dataframe, instead of implemented all the code
        in this function (get_user_recommendation), you can create a new one (create_recommendation_dataframe)
        that will be called in this function.
    - You can add other arguments to the function if you need to.
    """
    # FIXME
    pass

if __name__ == '__main__':
    model_dict = {'KNN user based cosine':
                      {'algo': 'KNN', 'model_kwargs':
                          {'k': 40, 'min_k': 1, 'sim_options': {'user_based': False, 'name': 'cosine'}},
                       'from_surprise': True, 'test_size': 0.2},
                  'KNN user based pearson':
                      {'algo': 'KNN', 'model_kwargs':
                          {'k': 40, 'min_k': 1, 'sim_options': {'user_based': False, 'name': 'pearson'}},
                       'from_surprise': True, 'test_size': 0.2},
                  'NMF user based':
                      {'algo': 'NMF', 'model_kwargs':
                          {'n_factors': 15, 'n_epochs': 50, 'verbose': False},
                       'from_surprise': True, 'test_size': 0.2},
                  'SVD user based':
                      {'algo': 'SVD', 'model_kwargs': {}, 'from_surprise': True, 'test_size': 0.2}
                  }
    df_res, model_dict = benchmark(model_dict)
    print(df_res)
