import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader, accuracy
from util.MyRecommender import MyRecommender

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

print("My film recommender")
model = MyRecommender()
model.fit(train_matrix)

prediction = model.test(test_matrix)
rmse = MyRecommender.RMSE(prediction)

print("RMSE:",rmse)
print("MSE:",MyRecommender.MSE(prediction))
#model.selectHyperParameter(train_matrix,test_matrix)


print("SVD model")
reader = Reader(rating_scale = (1,5) )
train_data = Dataset.load_from_df(
    train[["user_id","movie_id","rating"]],
    reader
)
train_set = train_data.build_full_trainset()

test_set = list(
    test[['user_id', 'movie_id', 'rating']]
    .itertuples(index=False, name=None)
)

model1 = SVD()
model1.fit(train_set)
predictions = model1.test(test_set)
accuracy.rmse(predictions)
accuracy.mae(predictions)