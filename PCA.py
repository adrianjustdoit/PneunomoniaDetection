import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# 1. Dataset Sederhana
data = {'x': [1, 2, 3], 'y': [1, 2, 3]}
df = pd.DataFrame(data)

# Print data asli
print("Data Asli:")
print(df)
print("-" * 30)

# 2. Ekstraksi Fitur dengan PCA (Mereduksi jadi 1 dimensi)
# Secara default, library PCA di scikit-learn sudah melakukan "mean centering" 
# (pengurangan rata-rata) di belakang layar.
pca = PCA(n_components=1)
final_data = pca.fit_transform(df)

# Print hasil transformasi
print("Hasil Transformasi PCA (Final Data):")
result_df = pd.DataFrame(data=final_data, columns=['Principal Component 1'])
print(result_df.round(3))