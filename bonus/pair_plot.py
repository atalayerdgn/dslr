import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from feature_importance import analyze_feature_importance

class PairPlot:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.numerical_columns = self.data.select_dtypes(include=['number']).columns
    def calculate_feature_importance(self):
        numerical_cols = self.numerical_columns.tolist()
        if 'Index' in numerical_cols:
            numerical_cols.remove('Index')
        X = self.data[numerical_cols].dropna().values
        feature_importance, explained_variance_ratio, principal_components = analyze_feature_importance(X, numerical_cols, num_components=3)
        return feature_importance, explained_variance_ratio, principal_components
    def pair_plot(self):
        plot_columns = [col for col in self.numerical_columns if col != 'Index']
        n_features = len(plot_columns)
        group_size = (n_features + 2) // 3 
        groups = []
        for i in range(0, n_features, group_size):
            groups.append(plot_columns[i:i+group_size])
        for i, group in enumerate(groups, 1):
            g = sns.pairplot(
                self.data[group], 
                diag_kind="hist",
                plot_kws={'alpha': 0.7, 's': 4},
                diag_kws={'bins': 15}
            )
            plt.show()
            if i < len(groups):
                input(f"Press Enter to continue to Group {i+1}...")

def main():
    pair_plot = PairPlot('datasets/dataset_train.csv')
    pair_plot.calculate_feature_importance()
    pair_plot.pair_plot()
    

if __name__ == "__main__":
    main()
