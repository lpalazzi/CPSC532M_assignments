# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier


# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1,16) # depths to try
     
        train_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            train_errors[i] = np.mean(y_pred != y)
        
        plt.plot(depths, train_errors, label="training error")

        print ("training errors = " + str(train_errors))

        test_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X_test)
            test_errors[i] = np.mean(y_pred != y_test)
        
        plt.plot(depths, test_errors, label="testing error")

        print ("testing errors = " + str(test_errors))

        # save figure
        filename = "q1_1_test_train_errors.pdf"
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", filename)
        plt.savefig(fname)
        print ("\nFigure saved as '%s'" % filename)



    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        depths = np.arange(1,15) # depths to try

        X_train, y_train = X[:int(n/2),:], y[:int(n/2)]
        X_val, y_val = X[int(n/2):,:], y[int(n/2):]

        best_error = 1
        best_depth = 0
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            val_error = np.mean(y_pred != y_val)
            if val_error < best_error:
                best_error = val_error
                best_depth = max_depth

        print ("Validation set 1: best error is " + str(best_error) + " at depth " + str(best_depth))

        X_val, y_val = X[:int(n/2),:], y[:int(n/2)]
        X_train, y_train = X[int(n/2):,:], y[int(n/2):]

        best_error = 1
        best_depth = 0
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            val_error = np.mean(y_pred != y_val)
            if val_error < best_error:
                best_error = val_error
                best_depth = max_depth

        print ("Validation set 2: best error is " + str(best_error) + " at depth " + str(best_depth))


    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        print ("1. wordlist[50] = " + str(wordlist[50]))
        print ("2. words present = " + str(wordlist[X[500,:] == 1]))
        print ("3. newsgroup name = " + str(groupnames[y[500]]))


    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model = BernoulliNB()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("scikit-learn's validation error: %.3f" % v_error)
    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        # ---------------
        # 3.2
        # ---------------

        for k in [1,3,10]:
            model = KNN(k)
            model.fit(X,y) 
            ytrain = model.predict(X)
            ypred = model.predict(Xtest)
            error_train = np.mean(y != ytrain)
            error_test  = np.mean(ypred != ytest)
            print ("\nk = " + str(k))
            print ("\ttraining error = " + str(error_train))
            print ("\ttesting error = " + str(error_test))

        # ---------------
        # 3.3
        # ---------------

        # plot for KNN implementation
        model = KNN(1)
        model.fit(X,y)
        utils.plotClassifier(model,model.X,model.y)
        filename = "q3_KNN.pdf"
        fname = os.path.join("..", "figs", filename)
        plt.savefig(fname)
        print ("\nFigure saved as '%s'" % filename)

        # plot for scikit-learn  implementation
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X,y)
        utils.plotClassifier(model,X,y)
        filename = "q3_sklearn.pdf"
        fname = os.path.join("..", "figs", filename)
        plt.savefig(fname)
        print ("\nFigure saved as '%s'" % filename)

    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)
        
        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree errors")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

        print("\nRandom tree errors")
        evaluate_model(RandomTree(max_depth=np.inf))

        # ---------------
        # 4.2, 4.3
        # ---------------
        t = time.time()
        print("\nRandom forest errors")
        evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))
        print("    Time taken: %f seconds" % (time.time()-t))

        # ---------------
        # 4.4
        # ---------------
        t = time.time()
        print("\nsciki-learn random forest errors")
        evaluate_model(RandomForestClassifier(n_estimators=50, max_depth=None))
        print("    Time taken: %f seconds" % (time.time()-t))


    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        
        # ---------------
        # 5.2
        # ---------------
        print ("\nQuestion 5.1.2")

        model = Kmeans(k=4)
        model.fit(X)
        model.print_errors()

        # ---------------
        # 5.3
        # ---------------
        print ("\nQuestion 5.1.3")

        models = np.array([Kmeans(k=4) for _ in range(50)])
        errors = np.zeros(50)
        for z in range(50):
            models[z].fit(X)
            errors[z] = models[z].error(X)

        best_model = models[np.argmin(errors)]
        y = best_model.predict(X)

        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")
        fname = os.path.join("..", "figs", "q5_1_best_model.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']
        
        errors = np.zeros(10)
        k_s = np.array(range(1,11))
        for k in k_s:
            errors_k = np.zeros(50)
            for z in range(50):
                model = Kmeans(k=k)
                model.fit(X)
                errors_k[z] = model.error(X)

            errors[k-1] = np.min(errors_k)
        
        # save figure
        plt.plot(k_s, errors)
        filename = "q5_2_min_errors.pdf"
        plt.xlabel("k")
        plt.ylabel("Minimum error")
        fname = os.path.join("..", "figs", filename)
        plt.savefig(fname)
        print ("\nFigure saved as '%s'" % filename)


    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=3, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
