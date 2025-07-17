import numpy as np

def normalize(X):
    N, D = X.shape
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    # Avoid division by zero
    sigma[sigma == 0] = 1
    
    # Standardize: (X - mean) / std
    X_normalized = (X - mu) / sigma
    
    return X_normalized, mu, sigma

def eig(S):
    eigvals, eigvecs = np.linalg.eig(S)
    sort_indices = np.argsort(eigvals)[::-1]
    return eigvals[sort_indices], eigvecs[:, sort_indices]

def projection_matrix(B):
    P = B @ np.linalg.inv(B.T @ B) @ B.T 
    return np.eye(B.shape[0]) @ P

def calculate_feature_importance(principal_components, principal_values, feature_names=None):
    """
    Calculate feature importance based on PCA components
    """
    total_variance = np.sum(principal_values)
    explained_variance_ratio = principal_values / total_variance
    feature_importance = np.zeros(principal_components.shape[0])
    
    for i, (component, variance_ratio) in enumerate(zip(principal_components.T, explained_variance_ratio)):
        weighted_loadings = np.abs(component) * variance_ratio
        feature_importance += weighted_loadings
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
    
    return feature_importance, explained_variance_ratio, feature_names

def print_feature_importance(feature_importance, feature_names):
    """
    Print top N most important features
    """
    sorted_indices = np.argsort(feature_importance)[::-1]
    print("Feature Importance Ranking:")
    print("-" * 50)
    for i, idx in enumerate(sorted_indices[:len(feature_names)]):
        print(f"{i+1:2d}. {feature_names[idx]:<30} {feature_importance[idx]:.6f}")

def PCA(X, num_components):
    X_normalized, mean, sigma = normalize(X)
    S = np.cov(X_normalized, rowvar=False, bias=True)
    eig_vals, eig_vecs = eig(S)
    principal_vals, principal_components = eig_vals[:num_components] ,eig_vecs[:, :num_components]
    principal_components = np.real(principal_components) 
    P = projection_matrix(eig_vecs[:,:num_components])
    x_reconst = P@X_normalized.T
    reconst = x_reconst.T * sigma + mean
    return reconst, mean, principal_vals, principal_components

def PCA_high_dim(X, num_components):
    N, D = X.shape
    X_normalized, mean, sigma = normalize(X)
    M = np.dot(X_normalized, X_normalized.T) / N
    eig_vals, eig_vecs = eig(M)
    eig_vecs = X_normalized.T @ eig_vecs
    principal_values = eig_vals[:num_components]
    principal_components = eig_vecs[:, :num_components]
    principal_components = np.real(principal_components)
    print(projection_matrix(principal_components).shape)
    reconst = (projection_matrix(principal_components) @ X_normalized.T).T * sigma + mean
    return reconst, mean, principal_values, principal_components

def analyze_feature_importance(X, feature_names=None, num_components=5):
    reconst, mean, principal_vals, principal_components = PCA(X, num_components)
    
    feature_importance, explained_variance_ratio, feature_names = calculate_feature_importance(
        principal_components, principal_vals, feature_names
    )
    print_feature_importance(feature_importance, feature_names)
    
    return feature_importance, explained_variance_ratio, principal_components
