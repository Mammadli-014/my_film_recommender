import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader, accuracy
from util import MyRecommender

ratings = pd.read_csv(
    r"C:\Users\user\Downloads\ml-100k\ml-100k\u.data",
    sep="\t",
    names=["user_id" , "movie_id", "rating" , "timestamp"]
)

train,test = train_test_split(ratings, test_size=0.2, random_state=42)
all_users = ratings["user_id"].unique()
all_movies = ratings["movie_id"].unique()

train_matrix = train.pivot(index="user_id",columns="movie_id",values="rating").reindex(index=all_users,columns=all_movies)
test_matrix = test.pivot(index="user_id",columns="movie_id",values="rating").reindex(index=all_users,columns=all_movies)


model = MyRecommender()
model.fit(train_matrix)

prediction = model.test(test_matrix)
rmse = MyRecommender.calculate_RMSE(prediction)

