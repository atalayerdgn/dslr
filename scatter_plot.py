import matplotlib.pyplot as plt
import pandas as pd

class ScatterPlot:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path)
        self.numerical_columns = self.data.select_dtypes(include=['number']).columns
    
    def calculate_mean(self, X):
        return sum(X) / len(X)
    
    def calculate_covariance(self, X, Y):
        result = 0
        for i in range(len(X)):
            result += (X[i] - self.calculate_mean(X)) * (Y[i] - self.calculate_mean(Y))
        return result / (len(X) - 1)
    
    def calculate_correlation(self, X, Y):
        return self.calculate_covariance(X, Y) / (self.calculate_std(X) * self.calculate_std(Y))
    
    def calculate_std(self, X):
        return (self.calculate_variance(X)) ** 0.5
    
    def calculate_variance(self, X):
        return sum((x - self.calculate_mean(X)) ** 2 for x in X) / (len(X) - 1)
    
    def pearson_correlation(self, X, Y):
        return self.calculate_correlation(X, Y)
    def plot_scatter(self, col1, col2):
        series1 = self.data[col1].dropna()
        series2 = self.data[col2].dropna()
        common_indices = series1.index.intersection(series2.index)
        
        if len(common_indices) > 1:
            X = series1[common_indices]
            Y = series2[common_indices]
            
            plt.figure(figsize=(8, 6))
            plt.scatter(X, Y, alpha=0.6)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f'Scatter Plot: {col1} vs {col2}')
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            print(f"Not enough data points for {col1} and {col2}")

def main():
    scatter_plot = ScatterPlot('datasets/dataset_train.csv')
    data = scatter_plot.numerical_columns
    corrs = []
    for i in data:
        for j in data:
            if i != j:
                series1 = scatter_plot.data[i].dropna()
                series2 = scatter_plot.data[j].dropna()
                common_indices = series1.index.intersection(series2.index)
                
                if len(common_indices) > 1:
                    X = series1[common_indices].tolist()
                    Y = series2[common_indices].tolist()
                    if len(set(X)) > 1 and len(set(Y)) > 1:
                        corr = scatter_plot.pearson_correlation(X, Y)
                        corrs.append((i, j, corr))
    corrs.sort(key=lambda x: x[2], reverse=True)
    for i, (col1, col2, corr) in enumerate(corrs[:10]):
        print(f"{i+1}. {col1} & {col2}: {corr:.6f}")
    if corrs:
        most_similar = corrs[0]
        print(f"{most_similar[0]} and {most_similar[1]} with correlation: {most_similar[2]:.6f}")
        scatter_plot.plot_scatter(most_similar[0], most_similar[1])

if __name__ == "__main__":
    main()
