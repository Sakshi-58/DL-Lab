# Perform PCA in dimension reduction of numerical data  
# a. Pre-process the data through standardization. 
# b. Perform PCA to reduce dimension.  
# c. Construct the scree plot. 
# d. Data visualization in lower dimensional representation. 



# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import numpy as np

# df = pd.read_csv("/content/iris.csv")
# df
# df.isnull().sum()

# df.info()

# X = data.select_dtypes(include=['int64', 'float64'])

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# pca = PCA()
# X_pca = pca.fit_transform(X_scaled)

# plt.scatter(X_pca[:, 0], X_pca[:, 1])
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA Visualization")
# plt.show()

# plt.plot(pca.explained_variance_ratio_, marker='o')
# plt.xlabel("Principal Components")
# plt.ylabel("Explained Variance Ratio")
# plt.title("Scree Plot")
# plt.show()














# # Perform PCA in dimension reduction of numerical data  
# # a. Pre-process the data through standardization. 
# # b. Perform PCA to reduce dimension.  
# # c. Construct the scree plot. 
# # d. Data visualization in lower dimensional representation. 



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


from sklearn.datasets import load_iris


# Load preloaded Iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(lambda x: iris.target_names[x])

# Load data
# df = pd.read_csv("/content/iris.csv")
df = pd.read_csv("iris.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())

# a. Pre-process: Select numerical columns and standardize
X = df.select_dtypes(include=['int64', 'float64'])  # Fixed: was 'data' (undefined)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# b. Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# c. Scree Plot
plt.figure(figsize=(8, 5))
components = range(1, len(pca.explained_variance_ratio_) + 1)  # Fixed: 1-based index
plt.plot(components, pca.explained_variance_ratio_, marker='o', label='Individual Variance')
plt.plot(components, np.cumsum(pca.explained_variance_ratio_), marker='s', linestyle='--', label='Cumulative Variance')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.xticks(components)
plt.legend()
plt.grid(True)
plt.show()

# d. 2D Visualization (colored by species if column exists)
plt.figure(figsize=(8, 5))
if 'species' in df.columns:
    species = df['species'].astype('category')
    colors = species.cat.codes
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='Set1')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(species.cat.categories), title="Species")
else:
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - 2D Visualization")
plt.grid(True)
plt.show()




















