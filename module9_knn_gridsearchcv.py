from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
def read_input_pairs(n, prompt=""):
  pairs = []
  for i in range(n):
    x = float(input(f"{prompt}Enter x for pair {i+1}: "))
    y = int(input(f{prompt}Enter y for pair {i+1}: "))
    pairs.append((x,y))
  return np.array(pairs)
def main():
    N = int(input("Enter number of training samples (N): "))
    train_data = read_input_pairs(N, prompt="Train - ")
    TrainX = train_data[:, 0].reshape (-1, 1)
    TrainY = train_data[:, 1]
    M = int(input("Enter number of test samples (M): "))
    test_data = read_input_pairs(M, prompt="Test - ")
    TestX = test_data[:,0].reshape(-1,1)
    TestY = test_data[:, 1]
    k_values = list(range(1, min(11, len(TrainX) + 1)))
    param_grid = {'n_neighbors': k_values}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 2)
    grid.fit(TrainX, TrainY)
    preds = grid.predict(TestX)
    accuracy = accuracy_score(TestY, preds)
    best_k = grid.best_params_['n_neighbors']
    print(f"\nBest k: {best_k}")
    print(f"Test set accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
    
    
