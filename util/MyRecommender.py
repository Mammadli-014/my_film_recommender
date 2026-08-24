import numpy as np
import pandas as pd

class MyRecommender:

    def __init__(self):
        self.global_mean = 0
        self.all_users = 0
        self.all_movies = 0
        self.W = None
        self.X = None
        self.B_film = None
        self.B_user = None


    def calculate_Global_Mean(self, Y, R):
        global_mean = np.sum(Y) / np.sum(R)
        return global_mean

    def predict(self, W, X, B_film, B_user):
        prediction_matrix = W @ X.T + B_user[:, np.newaxis] + B_film[np.newaxis, :] + self.global_mean
        return prediction_matrix


    def gradientDescent(self, W, X, B_film, B_user, yArray, rArray, iteration, learning_rate, momentum=0.8,lambda_=0.1):

        dW = None
        dX = None
        dB_film = None
        dB_user = None

        vW = np.zeros_like(W)
        vX = np.zeros_like(X)
        vB_film = np.zeros_like(B_film)
        vB_user = np.zeros_like(B_user)

        for i in range(iteration + 1):

            prediction = self.predict(W, X, B_film, B_user)
            error = (prediction - yArray) * rArray

            dW = np.dot(error, X) + lambda_ * W
            dX = np.dot(error.T, W) + lambda_ * X

            dB_film = np.sum(error, axis=0) + lambda_ * B_film
            dB_user = np.sum(error, axis=1) + lambda_ * B_user

            vW = momentum * vW + learning_rate * dW
            vX = momentum * vX + learning_rate * dX
            vB_film = momentum * vB_film + learning_rate * dB_film
            vB_user = momentum * vB_user + learning_rate * dB_user

            W = W - vW
            X = X - vX
            B_film = B_film - vB_film
            B_user = B_user - vB_user

            if i % 100 == 0:
                data_loss = np.sum(np.square(error) / 2)
                regularization_loss = (lambda_ / 2) * (np.sum(np.square(W)) + np.sum(np.square(X)))
                total_loss = data_loss + regularization_loss
                print(i,"th iteration     Loss:", "{:.4f}".format(total_loss))

        return W, X, B_film, B_user

    @staticmethod
    def RMSE(prediction):
        error = ( prediction["prediction"] - prediction["actual"] ) * prediction["mask"]
        n = np.sum(prediction["mask"])
        rmse = np.sqrt( np.sum(error ** 2) / n)
        return "{:.4f}".format(rmse)
    @staticmethod
    def MSE(prediction):
        error = ( prediction["prediction"] - prediction["actual"] ) * prediction["mask"]
        n = np.sum(prediction["mask"])
        mse = np.sum(np.square(error)) /n
        return "{:.4f}".format(mse)

    def split_Y_R(self, matrix):
        Y = np.array(matrix.copy())
        R = Y.copy()

        nan_mask = np.isnan(Y)
        Y[nan_mask] = 0
        R[nan_mask] = 0
        R[~nan_mask] = 1

        return Y, R

    def fit(self, train_set , feature_num = 3):
        Y_train , R_train = self.split_Y_R(train_set)
        self.all_users = Y_train.shape[0]
        self.all_movies = Y_train.shape[1]
        self.global_mean = self.calculate_Global_Mean(Y_train, R_train)

        W = np.random.randn(self.all_users, feature_num) * 0.01
        X = np.random.randn(self.all_movies, feature_num) * 0.01
        B_film = np.zeros((self.all_movies,))
        B_user = np.zeros((self.all_users,))

        self.W, self.X, self.B_film, self.B_user = (self.gradientDescent
        (W, X, B_film, B_user, Y_train, R_train, iteration=400, learning_rate=0.005, momentum=0.8, lambda_= 1))

    def test(self,test_matrix):
        Y , R =self.split_Y_R(test_matrix)
        prediction = self.predict(self.W, self.X, self.B_film, self.B_user)
        test_results = {
            "prediction": prediction,
            "actual": Y,
            "mask":R
        }
        return test_results

    def _testHyperParam(self, W, X, B_film, B_user , test_matrix):
        Y, R = self.split_Y_R(test_matrix)
        prediction = (W @ X.T + B_user[:, np.newaxis] + B_film[np.newaxis, :] + self.global_mean)
        test_results = {
            "prediction": prediction,
            "actual": Y,
            "mask": R
        }
        return test_results

    def selectHyperParameter(self,train_matrix, test_matrix):
        np.random.seed(42)
        learingRateList = [0.005, 0.001, 0.002]
        featureList = [3, 5, 7]
        lambdaList = [1, 0.5]
        print("Feature | Rate   | Lambda  | Train RMSE | Test RMSE")
        print("-----------------------------------------------------")

        for i in learingRateList:
            for j in featureList:
                for l in lambdaList:
                    feature_num = j
                    W = np.random.randn(self.all_users, feature_num) * 0.01
                    X = np.random.randn(self.all_movies, feature_num) * 0.01
                    B_film = np.zeros((self.all_movies,))
                    B_user = np.zeros((self.all_users,))
                    Y_train,R_train = self.split_Y_R(train_matrix)
                    W, X, B_film, B_user = (
                    self.gradientDescent(W, X, B_film, B_user,Y_train, R_train,
                                       iteration=400, learning_rate=i, momentum=0.85, lambda_= l))

                    predictionTrain = self._testHyperParam(W, X, B_film, B_user, train_matrix)
                    errorTrain = self.RMSE(predictionTrain)
                    predictionTest = self._testHyperParam(W, X, B_film, B_user, test_matrix)
                    errorTest = self.RMSE(predictionTest)

                    print(str(j).ljust(7), "| ",
                          str(i).ljust(7), "| ",
                          str(l).ljust(7), "| ",
                          str(errorTrain).ljust(3), "| ",
                          str(errorTest).ljust(3))