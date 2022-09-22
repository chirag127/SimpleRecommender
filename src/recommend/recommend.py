"""
function for recommendation
"""
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_table(list_of_movie_enjoyed, tfidf_data, movie_count=20):
    """
    function for recommending movies
    :param list_of_movie_enjoyed: list of movies
    :param tfidf_data: self-explanatory
    :param movie_count: no of movies to suggest
    :return: dataframe containing suggested movie
    """
    movie_enjoyed_df = tfidf_data.reindex(list_of_movie_enjoyed)
    user_prof = movie_enjoyed_df.mean()
    tfidf_subset_df = tfidf_data.drop(list_of_movie_enjoyed)
    similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
    similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])
    return similarity_df.sort_values(
        by="similarity_score", ascending=False
    ).head(movie_count)
