# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    print("La probabilité que la plante soit toxique ", A)
    
    return A >= 0.5

def artificial_neuron(X, y, learning_rate=0.1, n_iter=100):
    
        W, b = initialisation(X)

        Loss = []

        for i in range(n_iter):
            A = model(X, W, b)
            Loss.append(log_loss(A, y))
            dW, db = gradients(A, X, y)
            W, b = update(dW, db, W, b, learning_rate)

        
        # Sauvegarde des poids et des biais pour une utilisation ultérieure
        #np.savez('W_b_training.npz', W=W, b=b)
        
        y_pred = predict(X, W, b)
        print("Précision des prédictions on training data:", accuracy_score(y, y_pred))

        #return W, b



def main():
    
   
    # Charger les poids et les biais depuis le fichier
    loaded_model = np.load('W_b_training.npz')
    W_loaded, b_loaded = loaded_model['W'], loaded_model['b']
    
    # Génération des données d'entraînement
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X[:,0], X[:, 1], c=y.flatten(), cmap='summer')

    x1 = np.linspace(-1, 4, 100)
    x2 = ( - W_loaded[0] * x1 - b_loaded) / W_loaded[1]

    ax.plot(x1, x2, c='orange', lw=3)
    
    predictions = predict(X, W_loaded, b_loaded)

    # Comparer les prédictions avec les vraies étiquettes
    accuracy = accuracy_score(y, predictions)
    print("Précision des prédictions on the training data :", accuracy)


    # Entraînement du modèle
    # artificial_neuron(X, y) # training already done 
    
    
    # Prediction de nouvelles plantes 
    new_plant = np.array([2,1])
    plt.scatter(new_plant[0], new_plant[1], c='r')
    
    predictions = predict(new_plant, W_loaded, b_loaded)
    print("La plante est-elle toxique :", predictions)
    
    new_plant1 = np.array([3,5])
    plt.scatter(new_plant1[0], new_plant1[1], c='r')
    
    predictions = predict(new_plant1, W_loaded, b_loaded)
    print("La plante est-elle toxique :", predictions)
    
    new_plant2 = np.array([1,2])
    plt.scatter(new_plant2[0], new_plant2[1], c='r')
    
    predictions = predict(new_plant2, W_loaded, b_loaded)
    print("La plante est-elle toxique :", predictions)
    

if __name__ == "__main__":
    main()

