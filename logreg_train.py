import numpy as np
import pandas as pd
import json

class LogRegTrain:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.numerical_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        self.numerical_cols.remove('Index')
        self.numerical_cols.remove('Arithmancy')
        self.numerical_cols.remove('Care of Magical Creatures')
        
        self.X = self.data[self.numerical_cols].fillna(self.data[self.numerical_cols].mean())
        self.X_mean = self.X.mean()
        self.X_std = self.X.std()
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        self.houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        self.y_original = self.data['Hogwarts House']
        
        bias_col = np.ones((self.X_normalized.shape[0], 1))
        self.X2 = np.concatenate((bias_col, self.X_normalized.values), axis=1)
        
        self.weights = {}
        for house in self.houses:
            self.weights[house] = np.zeros((self.X2.shape[1], 1))
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def calc_cost(self, theta, X, y):
        m = len(y)
        z = np.dot(X, theta)
        y_hat = self.sigmoid(z)
        y = y.reshape(-1, 1)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        term1 = y * np.log(y_hat)
        term2 = (1 - y) * np.log(1 - y_hat)
        J = -np.sum(term1 + term2) / m
        grad = np.dot(X.T, (y_hat - y)) / m
        return (J, grad)
    
    def create_binary(self, target_house):
        return (self.y_original == target_house).astype(int)
    
    def train_one(self, house, learning_rate=0.1, max_iterations=1000):
        y_binary = self.create_binary(house)
        theta = self.weights[house].copy()
        costs = []
        for i in range(max_iterations):
            cost, grad = self.calc_cost(theta, self.X2, np.array(y_binary))
            costs.append(cost)
            theta = theta - learning_rate * grad
            if i % 200 == 0:
                print(f"  Iteration {i}: Cost = {cost:.6f}")
        self.weights[house] = theta
        return theta, costs
    
    def train(self, learning_rate=0.1, max_iterations=1000):
        all_costs = {}
        for house in self.houses:
            theta, costs = self.train_one(house, learning_rate, max_iterations)
            all_costs[house] = costs
        return all_costs
    
    def save_weights(self, filename="weights.json"):
        weights_dict = {
            'houses': self.houses,
            'weights': {},
            'feature_names': self.numerical_cols,
            'normalization_params': {
                'mean': self.X_mean.tolist(),
                'std': self.X_std.tolist()
            },
            'metadata': {
                'features_count': len(self.numerical_cols),
                'training_samples': self.X2.shape[0],
                'model_type': 'Logistic Regression'
            }
        }
        
        for house in self.houses:
            weights_dict['weights'][house] = {
                'bias': float(self.weights[house][0, 0]),
                'coefficients': self.weights[house][1:].flatten().tolist()
            }
        
        with open(filename, 'w') as f:
            json.dump(weights_dict, f, indent=2)
        return filename

if __name__ == "__main__":
    l = LogRegTrain('datasets/dataset_train.csv')
    l.train(0.1, 1000)
    l.save_weights("weights.json")
