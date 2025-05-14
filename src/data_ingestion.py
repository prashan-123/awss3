import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

df=pd.read_csv(r"P:\awss3\data\raw\student_placement_data.csv")
X=df.drop(columns=['PLACED'])
y=df['PLACED']

ss=StandardScaler()
X_scaled=ss.fit_transform(X)

pca=PCA(n_components=3)
X_pca=pca.fit_transform(X_scaled)

df_pca=pd.DataFrame(data=X_pca,columns=["PC1","PC2","PC3"])
df_pca["Placed"]=y.values

df_pca.to_csv(os.path.join('data','processed','student_perfromance_pca.csv'),index=False)