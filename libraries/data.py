from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Data():
    def __init__(self, data_csv: str, pca_components: int) -> None:
        self.data = pd.read_csv(data_csv)
        self.data.columns = [*self.data.columns[:-1], 'Target Class']

        self.scaler = StandardScaler()
        self.scaler.fit(self.data.drop('Target Class', axis=1).values)
        self.scaled_data = self.scaler.transform(self.data.drop('Target Class',axis=1).values)

        self.pca = PCA(n_components=pca_components)
        self.pca.fit(self.scaled_data)