import math
import pandas as pd # for testing


class Describe:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.columns = []
    
    def read_data(self):
        with open(self.path, 'r') as file:
            lines = file.readlines()
        
        if lines:
            self.columns = lines[0].strip().split(',')
            
            # Initialize data dictionary
            for col in self.columns:
                self.data[col] = []
            
            # Parse data rows
            for line in lines[1:]:
                if line.strip():
                    row = line.strip().split(',')
                    for i, col in enumerate(self.columns):
                        if i < len(row):
                            # Try to convert to float for numerical data
                            try:
                                self.data[col].append(float(row[i]))
                            except ValueError:
                                # Keep as string for non-numerical data
                                self.data[col].append(row[i])
    
    def count_data(self, column_name):
        return len(self.data[column_name])
    
    def mean_data(self, column_name):
        numerical_values = [x for x in self.data[column_name] if isinstance(x, (int, float))]
        if not numerical_values:
            return None
        return sum(numerical_values) / len(numerical_values)
    
    def std_data(self, column_name):
        numerical_values = [x for x in self.data[column_name] if isinstance(x, (int, float))]
        if not numerical_values:
            return None
        mean_val = sum(numerical_values) / len(numerical_values)
        return math.sqrt(sum((x - mean_val) ** 2 for x in numerical_values) / len(numerical_values))
    
    def min_data(self, column_name):
        numerical_values = [x for x in self.data[column_name] if isinstance(x, (int, float))]
        if not numerical_values:
            return None
        min_val = numerical_values[0]
        for x in numerical_values:
            if x < min_val:
                min_val = x
        return min_val
    
    def max_data(self, column_name):
        numerical_values = [x for x in self.data[column_name] if isinstance(x, (int, float))]
        if not numerical_values:
            return None
        max_val = numerical_values[0]
        for x in numerical_values:
            if x > max_val:
                max_val = x
        return max_val
    
    def percentile_data(self, column_name, percentile):
        numerical_values = [x for x in self.data[column_name] if isinstance(x, (int, float))]
        if not numerical_values:
            return None
        sorted_values = sorted(numerical_values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = lower_index + 1
        if upper_index > len(sorted_values) - 1:
            upper_index = len(sorted_values) - 1
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def q1_data(self, column_name):
        return self.percentile_data(column_name, 25)
    
    def median_data(self, column_name):
        return self.percentile_data(column_name, 50)
    
    def q3_data(self, column_name):
        return self.percentile_data(column_name, 75)

    def describe_data(self):
        for column_name in self.columns:
            # Check if column has numerical data
            numerical_values = [x for x in self.data[column_name] if isinstance(x, (int, float))]
            if numerical_values:
                print(f"Column: {column_name}")
                print(f"Count: {len(numerical_values)}")
                print(f"Mean: {self.mean_data(column_name)}")
                print(f"Std: {self.std_data(column_name)}")
                print(f"Min: {self.min_data(column_name)}")
                print(f"25%: {self.q1_data(column_name)}")
                print(f"50%: {self.median_data(column_name)}")
                print(f"75%: {self.q3_data(column_name)}")
                print(f"Max: {self.max_data(column_name)}")
                print("-" * 30)

def main():
    describe = Describe('datasets/dataset_train.csv')
    describe.read_data()
    print("my data:")
    describe.describe_data()
    print("pandas data:")
    print(pd.read_csv('datasets/dataset_train.csv').describe())

if __name__ == "__main__":
    main()


