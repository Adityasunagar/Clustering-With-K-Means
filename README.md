K-Means Clustering – Customer Segmentation
  --This project performs unsupervised learning using K-Means clustering on a Customer Segmentation dataset.
  --The goal is to group customers into clusters based on income and spending behavior.

1. Objective
  --Load and preprocess customer dataset
  --Apply K-Means clustering
  --Used the Elbow Method to find optimal K
  --Visualize clusters with Matplotlib
  --Evaluate clustering using the Silhouette Score

2. Dataset
  --Dataset used: Mall Customer Segmentation Dataset
  --Source (Kaggle):
      https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

*Common columns:

    CustomerID

    Gender

    Age

    Annual Income (k$)

    Spending Score (1–100)

*Popular features for clustering:

    Annual Income
    Spending Score

3. Tools Used

    Python
    Pandas
    NumPy
    Matplotlib
    scikit-learn
    Jupyter Notebook (optional)

4. Import Required Libraries
  """   import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA   """

5. Project Workflow
    Step 1: Load the dataset & Standardize features
  
    Step 2: Use Elbow Method to choose K

    Step 3: Fit K-Means model (example: K=5)

    Step 4: Visualization Clusters (2D)
    <img width="695" height="547" alt="67fa5f79-59cb-403c-ba0f-d134f1105286" src="https://github.com/user-attachments/assets/783a28bc-e1c0-4949-8dfd-69e3e8c44acf" />

    Step 5: PCA Visualization (2D)
   
    <img width="689" height="547" alt="5469329c-7296-4300-9e4d-8dfa7910b477" src="https://github.com/user-attachments/assets/ecb2a03c-57af-4ec4-a220-c4c6ce024f55" />

    Step 6: Evaluate with Silhouette Score
          --Silhouette Score: 0.5546571631111091

7. PCA Explanation 

    Principal Component Analysis (PCA) reduces high-dimensional data into 2D or 3D for visualization.

    K-Means works better when dimensions are reduced and Removes noise

    Helps visualize clusters in 2D even if original dataset has many features

Example:

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
    plt.title("Clusters after PCA Reduction")

8. Silhouette Score Explanation

    Silhouette Score measures how well clusters are separated.

    Range:

        1.0 → excellent clustering

        0.5–0.7 → good clustering

        0.3–0.5 → average

        < 0.3 → poor clustering

        Negative → wrong clustering

  --Formula considers:

      Distance to points in the same cluster (cohesion)

      Distance to points in nearest cluster (separation)
