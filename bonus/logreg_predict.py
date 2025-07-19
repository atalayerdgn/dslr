import numpy as np
import pandas as pd
import json




class LogRegPredict:
    def __init__(self,w_path,test_path):
        self.weight = self.read_json(w_path)
        self.x_test = pd.read_csv(test_path)
        self.feature_names = self.weight['feature_names']
        self.houses = self.weight['houses']
        self.X = self.x_test[self.feature_names].fillna(self.x_test[self.feature_names].mean())
        self.X_mean = pd.Series(self.weight['normalization_params']['mean'], index=self.feature_names)
        self.X_std = pd.Series(self.weight['normalization_params']['std'], index=self.feature_names)
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        bias_col = np.ones((self.X_normalized.shape[0], 1))
        self.x_test = np.concatenate((bias_col, self.X_normalized.values), axis=1)
    def read_json(self,path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    def predict(self):    
        all_predictions = []
        for house in self.houses:
            bias = self.weight['weights'][house]['bias']
            coef = self.weight['weights'][house]['coefficients']
            theta = np.array([bias] + coef).reshape(-1, 1)
            
            # Use matrix multiplication (dot product) instead of element-wise multiplication
            z = np.dot(self.x_test, theta)
            predictions = self.sigmoid(z)
            all_predictions.append(predictions.flatten())
        
        # Convert to numpy array for easier manipulation
        all_predictions = np.array(all_predictions).T  # Shape: (n_samples, n_houses)
        
        # For each sample, predict the house with highest probability
        predicted_indices = np.argmax(all_predictions, axis=1)
        predicted_houses = [self.houses[i] for i in predicted_indices]
        
        # Create output DataFrame
        results = pd.DataFrame({
            'Index': range(len(predicted_houses)),
            'Hogwarts House': predicted_houses
        })
        
        # Save to CSV
        results.to_csv('houses.csv', index=False)
        print(f"Predictions saved to houses.csv")
        print(f"Predicted {len(predicted_houses)} houses")
        return results

def main():
    lr = LogRegPredict('weights.json','datasets/dataset_test.csv')
    lr.predict()
    
if __name__ == '__main__':
    main()
            
        
        
