import matplotlib.pyplot as plt
import pandas as pd
from describe import Describe
class Histogram:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path)
        self.numerical_columns = self.data.select_dtypes(include=['number']).columns
        self.variances = []
    def calc_variance(self, column_name):
        values = self.data[column_name]
        mean = values.mean()
        variance = ((values - mean) ** 2).mean()
        return variance
    def print_variance(self):
        numerical_columns = self.data.select_dtypes(include=['number']).columns
        self.variances = sorted([(self.calc_variance(column),column) for column in numerical_columns])
        print(self.variances)
        print(self.data[numerical_columns].std())
    def plot_histogram(self):
        columns = ['Divination','Defense Against the Dark Arts', 'Herbology', 'History of Magic']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, column in enumerate(columns):
            if column in self.data.columns:
                values = self.data[column].dropna()
                axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Histogram of {column}')
                axes[i].set_xlabel('Values')
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
def main():
    histogram = Histogram('datasets/dataset_train.csv')
    histogram.print_variance()
    histogram.plot_histogram()
if __name__ == "__main__":
    main()
        
