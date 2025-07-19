import numpy as np
import pandas as pd
import json

class LogRegTrain2:
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
    
    def calc_cost_single(self, theta, X_single, y_single):
        """Single sample cost and gradient calculation for SGD"""
        z = np.dot(X_single, theta)
        y_hat = self.sigmoid(z)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        # Cost for single sample
        J = -(y_single * np.log(y_hat) + (1 - y_single) * np.log(1 - y_hat))
        
        # Gradient for single sample  
        grad = np.dot(X_single.T, (y_hat - y_single))
        return (J[0], grad)

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
    
    def train_one_sgd(self, house, learning_rate=0.01, max_epochs=100):
        """True Stochastic Gradient Descent - one sample at a time"""
        y_binary = self.create_binary(house)
        theta = self.weights[house].copy()
        costs = []
        n_samples = len(y_binary)
        
        for epoch in range(max_epochs):
            epoch_cost = 0.0
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                # Process ONE sample at a time
                X_single = self.X2[idx:idx+1]
                y_single = y_binary[idx]
                
                cost, grad = self.calc_cost_single(theta, X_single, y_single)
                epoch_cost += cost

                theta = theta - learning_rate * grad
            
            avg_cost = float(epoch_cost / n_samples)
            costs.append(avg_cost)
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Avg Cost = {avg_cost:.6f}")
        
        self.weights[house] = theta
        return theta, costs

    def train_one_mini_batch(self, house, learning_rate=0.05, max_epochs=100, batch_size=32):
        """Mini-Batch Gradient Descent"""
        y_binary = self.create_binary(house)
        theta = self.weights[house].copy()
        costs = []
        n_samples = len(y_binary)
        
        for epoch in range(max_epochs):
            epoch_cost = 0.0
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            
            # Process in mini-batches
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = self.X2[batch_indices]
                y_batch = np.array(y_binary[batch_indices])
                
                cost, grad = self.calc_cost(theta, X_batch, y_batch)
                epoch_cost += cost * len(batch_indices)
                
                # Mini-batch update
                theta = theta - learning_rate * grad
            
            avg_cost = float(epoch_cost / n_samples)
            costs.append(avg_cost)
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Avg Cost = {avg_cost:.6f}")
        
        self.weights[house] = theta
        return theta, costs
    
    def train(self, method='batch', learning_rate=0.1, max_iterations=1000, batch_size=32):
        """
        Train using different gradient descent methods
        method: 'batch', 'sgd', or 'mini_batch'
        """
        all_costs = {}
        
        if method == 'batch':
            print("🔸 Using Batch Gradient Descent")
            for house in self.houses:
                print(f"Training {house}...")
                theta, costs = self.train_one(house, learning_rate, max_iterations)
                all_costs[house] = costs
                
        elif method == 'sgd':
            print("⚡ Using Stochastic Gradient Descent")
            for house in self.houses:
                print(f"Training {house}...")
                theta, costs = self.train_one_sgd(house, learning_rate, max_iterations)
                all_costs[house] = costs
                
        elif method == 'mini_batch':
            print("🔄 Using Mini-Batch Gradient Descent")
            for house in self.houses:
                print(f"Training {house}...")
                theta, costs = self.train_one_mini_batch(house, learning_rate, max_iterations, batch_size)
                all_costs[house] = costs
        else:
            raise ValueError("Method must be 'batch', 'sgd', or 'mini_batch'")
            
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
    import time
    
    print("=== Logistic Regression: Gradient Descent Comparison ===\n")
    
    # Test all three methods
    methods = [
        ('batch', 0.1, 200),      # Batch GD: higher learning rate, fewer iterations
        ('sgd', 0.01, 50),        # SGD: lower learning rate, fewer epochs  
        ('mini_batch', 0.05, 100) # Mini-batch: middle ground
    ]
    
    for method, lr, iterations in methods:
        print(f"\n{'='*50}")
        print(f"Testing {method.upper()} method")
        print(f"Learning Rate: {lr}, Iterations/Epochs: {iterations}")
        print(f"{'='*50}")
        
        start_time = time.time()
        l = LogRegTrain2('datasets/dataset_train.csv')
        all_costs = l.train(method=method, learning_rate=lr, max_iterations=iterations)
        end_time = time.time()
        
        print(f"\n⏱️ Training Time: {end_time - start_time:.2f} seconds")
        print(f"📊 Final Costs:")
        for house, costs in all_costs.items():
            print(f"  {house}: {costs[-1]:.6f}")
        
        # Save weights with method name
        weights_file = f"weights_{method}.json"
        l.save_weights(weights_file)
        print(f"💾 Weights saved to: {weights_file}")
    
    print(f"\n{'='*50}")
    print("🎯 Comparison Complete!")
    print("📝 Key Differences:")
    print("   • Batch GD: Stable, slower on large data")
    print("   • SGD: Fast, noisy convergence") 
    print("   • Mini-Batch: Best of both worlds")
    print(f"{'='*50}")
