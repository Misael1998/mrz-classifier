import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
from PIL import Image

class Cluster():

    def __init__(self, n_components = 50, n_clusters = 20, random_state = 42):
        self.pca = PCA(n_components=n_components)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)

    def create_clusters(self, features, clusters):
        df = pd.read_csv(features)
        features = df.drop(columns=["path"]).values

        features_pca = self.pca.fit_transform(features)

        labels = self.kmeans.fit_predict(features_pca)

        df["cluster"] = labels

        df.to_csv(clusters, index=False)
        print(f'✅ Clusters saved to {clusters}')
        return df

    def save_clusters_as_images(self, clusters, destination = 'clustered_images'):
        df = clusters
        os.makedirs(destination, exist_ok=True)

        for cluster_id in df["cluster"].unique():
            cluster_dir = os.path.join(destination, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)

        for _, row in df.iterrows():
            src_path = row["path"]
            cluster_id = row["cluster"]
            dst_dir = os.path.join(destination, f"cluster_{cluster_id}")

            try:
                img = Image.open(src_path).convert("RGB")
                filename = os.path.splitext(os.path.basename(src_path))[0] + ".jpg"
                dst_path = os.path.join(dst_dir, filename)
                img.save(dst_path, format="JPEG", quality=95)
            except Exception as e:
                print(f"❌ Error processing {src_path}: {e}")
