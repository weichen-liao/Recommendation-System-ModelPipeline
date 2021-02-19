# -*- coding: utf-8 -*-
# Author: Weichen Liao

import numpy as np
import pandas as pd
import pickle
from surprise.prediction_algorithms.algo_base import AlgoBase
from pipeline import get_data

def get_predict_data( user_id: int, data_path: str):
    '''
    :param user_id: the user_id
    :param data_path: the csv path of movie data
    :return: numpy array for prediction, basically an array of every movie in the dataset for user k
    '''
    df_movies = pd.read_csv(data_path)
    df_test = df_movies[['movieId']]
    df_test['userId'] = str(user_id)
    df_test['rating'] = np.nan
    df_test = df_test[['userId', 'movieId', 'rating']]
    df_test['movieId'] = df_test['movieId'].astype('str')
    return df_test.to_numpy(), df_movies

def recommendation_res(k: int, pred_res: list, movies: pd.DataFrame):
    res = []
    for item in pred_res:
        if item[4]['was_impossible'] == False:
            res.append([item[0], item[1], item[3]])
    res = pd.DataFrame(res, columns=['userId', 'movieId', 'pred_rating'])
    res = res.sort_values(by='pred_rating', ascending=False).reset_index(drop=True)
    res = res[:k]
    movies['movieId'] = movies['movieId'].astype('str')
    res = pd.merge(res[['movieId', 'pred_rating']], movies, how='left', on='movieId')

    return res[['movieId', 'title', 'genres', 'pred_rating']]

def get_user_recommendation(model_path: str, user_id: int, k: int, movie_path: str) -> pd.DataFrame:
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
        movie_path : str
            The path of csv containing the movies metadata (title, genre, etc)

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
    # load the model
    model = pickle.load(open(model_path, 'rb'))
    # generate the data for prediction
    test, df_movies = get_predict_data(user_id, movie_path)
    # make the predictions
    pred = model.test(test)
    # get recommendation result
    res = recommendation_res(k=k, pred_res=pred, movies=df_movies)
    return res

if __name__ == '__main__':
    data = get_data(from_surprise=True)
    res =get_user_recommendation(model_path='NMF.model', user_id=1, k=10, movie_path='../data/movielens/ml-small/movies.csv')
    print(res)