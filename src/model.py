import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('data/Customer.csv')

data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

x = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

n_clusters=5
model_1 = KMeans(n_clusters=n_clusters, init='k-means++').fit(x)

print(model_1.labels_)
plt.scatter(data['CustomerID'], x[:,3], c=model_1.labels_)
plt.show()