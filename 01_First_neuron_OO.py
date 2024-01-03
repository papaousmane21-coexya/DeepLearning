import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.W = None
        self.b = None

    def _initialize_parameters(self, X):
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def _model(self, X):
        Z = X.dot(self.W) + self.b
        A = self._sigmoid(Z)
        return A

    def _compute_log_loss(self, A, y):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    def _compute_gradients(self, A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return (dW, db)

    def _update_parameters(self, dW, db):
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db


    def _predict(self, X):
        A = self._model(X)
        prediction = A >= 0.5
        return (A, prediction)


    def fit(self, X, y):
        self._initialize_parameters(X)
        Loss = []

        for i in range(self.n_iter):
            A = self._model(X)
            Loss.append(self._compute_log_loss(A, y))
            dW, db = self._compute_gradients(A, X, y)
            self._update_parameters(dW, db)

        # Sauvegarde des poids et des biais pour une utilisation ultérieure
        np.savez('W_b_training_OO.npz', W=self.W, b=self.b)

        A_pred,predictions = self._predict(X)
        print("Précision des prédictions sur les données d'entraînement:", accuracy_score(y, predictions))
    
    def load_parameters(self, file_path):
        # Charger les poids et les biais depuis le fichier
        loaded_model = np.load(file_path)
        self.W, self.b = loaded_model['W'], loaded_model['b']


    '''def predict_new_data(self, X):
        A, predictions = self._predict(X)
        
        #if predictions.any():
        
        #pour afficher toutes les valeurs de notre tableau
        for i in range(len(predictions)):
            if predictions[i]:
                print("La probabilité que cette plante soit toxique est de :{}, alors elle l'est".format(A[i]))
            else:
                print("La probabilité que cette plante soit toxique est de :{}, alors elle ne l'est pas".format(A[i]))
         
        return predictions'''
    
    def predict_new_data(self, X):
        A, predictions = self._predict(X)
        if predictions.any():
            index_of_true = np.where(predictions)[0][0]  # Trouver l'index de la première prédiction True
            print("La probabilité que cette plante soit toxique est de :{}, alors elle l'est".format(A[index_of_true]))
        else:
            print("Aucune prédiction positive. La probabilité que cette plante soit toxique est de :{}, alors elle ne l'est pas".format(A[0]))
    
        return predictions


def main():
    # Génération des données d'entraînement
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    print('dimensions de X:', X.shape)
    print('dimensions de y:', y.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='summer')
    plt.show()

    #Entraînement du modèle
    #model = LogisticRegression()
    #model.fit(X, y)
    
    # Charger les poids et les biais depuis le fichier sans réentraîner
    model = LogisticRegression()
    model.load_parameters('W_b_training_OO.npz')
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X[:,0], X[:, 1], c=y.flatten(), cmap='summer')

    x1 = np.linspace(-2, 4, 100)
    x2 = ( - model.W[0] * x1 - model.b) / model.W[1]

    ax.plot(x1, x2, c='orange', lw=3)


    # Faire des prédictions avec le modèle entraîné
    predictions = model.predict_new_data(X)

    # Comparer les prédictions avec les vraies étiquettes
    accuracy = accuracy_score(y, predictions)
    print("Précision des prédictions sur les nouvelles données :", accuracy)
    
    # Prédictions sur de nouvelles plantes
    new_plant = np.array([2, 1])
    plt.scatter(new_plant[0], new_plant[1], c='r')
    predictions = model.predict_new_data(new_plant.reshape(1, -1))
    

    new_plant1 = np.array([3, 5])
    plt.scatter(new_plant1[0], new_plant1[1], c='r')
    predictions = model.predict_new_data(new_plant1.reshape(1, -1))
    

    new_plant2 = np.array([1, 2])
    plt.scatter(new_plant2[0], new_plant2[1], c='r')
    predictions = model.predict_new_data(new_plant2.reshape(1, -1))
   
    

    plt.show()

if __name__ == "__main__":
    main()
