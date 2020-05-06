
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        n = len(set(ratings["user"]))   # Get the amount of users and items. 
        d = len(set(ratings["item"]))   #
        arg = np.argmax(np.sum(X,axis=0))
        itemnumber = item_inverse_mapper[arg]
        print("1.1.1")
        print("The highest rated item is " + str(itemnumber))
        print("The rating of this item is " + str(np.sum(X,axis=0)[:,arg]))
        # YOUR CODE HERE FOR Q1.1.2
        print("1.1.2")
        user_entries = X.getnnz(axis=1)
        max_rater = np.argmax(user_entries)
        max_ratings = np.max(user_entries)
        print("The user " + str(user_inverse_mapper[max_rater]) + " has rated the most items.")
        print("This user has rated " + str(max_ratings) + " items. ")

        # YOUR CODE HERE FOR Q1.1.3
        # Number of ratings by user. 
        plt.yscale('log', nonposy = 'clip')
        plt.hist(user_entries, bins = 100)
        plt.xlabel("Number of items rated")
        plt.ylabel("Number of users")
        plt.show()
        # Number of ratings by item. 
        plt.figure()
        plt.yscale('log', nonposy = 'clip')
        plt.hist(X.getnnz(axis = 0), bins = 100)
        plt.xlabel("Number of ratings")
        plt.ylabel("Number of items")
        plt.show()
        # Plot of each rating. 
        plt.figure() 
        counter = np.array([np.sum(((np.sum(X == 1.0,axis = -1)))),
            np.sum(((np.sum(X == 2.0,axis = -1)))),
            np.sum(((np.sum(X == 3.0,axis = -1)))),
            np.sum(((np.sum(X == 4.0,axis = -1)))),
            np.sum(((np.sum(X == 5.0,axis = -1))))])
        print(counter)
        plt.bar(np.arange(1,6),counter)
        plt.ylabel("Number of ratings")
        plt.xlabel("Stars")
        plt.show()

    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]
        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        neighEu = NearestNeighbors()
        neighEu.fit(np.transpose(X))
        nearest_indices_euclidean = neighEu.kneighbors(X = np.transpose(grill_brush_vec),n_neighbors = 6)[1]
        nearest_items = [item_inverse_mapper[i] for i in nearest_indices_euclidean[0][1:6]]
        print("Q 1.2.1")
        print("URL for the nearest items: (euclidean) ")
        for i in range(5):
            print(url_amazon % nearest_items[i])
        X_normalized = normalize(X)
        neighNorm = NearestNeighbors()
        neighNorm.fit(np.transpose(X_normalized))
        nearest_indices_normal = neighNorm.kneighbors(X = np.transpose(grill_brush_vec),n_neighbors = 6)[1]
        nearest_items = [item_inverse_mapper[i] for i in nearest_indices_normal[0][1:6]]
        print("Q 1.2.2")
        print("URL for the nearest items: (normalized euclidean) ")
        for i in range(5):
            print(url_amazon % nearest_items[i])
        neighCos = NearestNeighbors(metric = 'cosine')
        neighCos.fit(np.transpose(X_normalized))
        nearest_indices_cosine = neighCos.kneighbors(X = np.transpose(grill_brush_vec),n_neighbors = 6)[1]
        nearest_items = [item_inverse_mapper[i] for i in nearest_indices_cosine[0][1:6]]
        print("Q 1.2.3")
        print("URL for the nearest items: (cosine) ")
        for i in range(5):
            print(url_amazon % nearest_items[i])
        # YOUR CODE HERE FOR Q1.3
        reviews_euclidean = [X[:,i].getnnz() for i in nearest_indices_euclidean[0][1:6]]
        reviews_cosine = [X[:,i].getnnz() for i in nearest_indices_cosine[0][1:6]]
        print("Number of ratings for the 5 nearest neighbors with euclidean metric: ")
        print(reviews_euclidean)
        print("Mean number of ratings for euclidean predictions: " + str(np.mean(reviews_euclidean)))
        print("Number of ratings for the 5 nearest neighbors with cosine metric: ")   
        print(reviews_cosine)     
        print("Mean number of ratings for cosine predictions: " + str(np.mean(reviews_cosine)))
        # reviews_euclidean = np.bincount(X[:,nearest_indices_euclidean[0][]].getnnz(axis=0))
        # print(reviews_euclidean)
        # print(item_inverse_mapper[nearest_indices_euclidean[0][1]])


    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']
        V = np.ones((500,1))
        V[400:500] = 0.1
        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,np.identity(len(V))*V)
        print(model.w)
        utils.test_and_plot(model,X,y,title="Weighted Least Squares",filename="weighted_least_squares_outliers.pdf")

        # YOUR CODE HERE

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        model = linear_model.LeastSquaresBias()
        model.fit(X,y)
        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, with bias",filename="q4-1_least_squares_with_bias.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, with bias",filename="q4-2_least_squares_poly_p"+str(p)+".pdf")
            print("\n")

    else:
        print("Unknown question: %s" % question)

