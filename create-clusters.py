import kagglehub
from models.features import Model
from models.clusters import Cluster

# Download latest version
path = kagglehub.dataset_download("prise6/earth-from-iss-hdev-experiment")

features_dir = './data/pre-train/features.csv'
clusters_dir = './data/pre-train/clusters.csv'
images_dir = './data/pre-train/clustered_images'

model = Model()
model.extract_features(path)
model.save_features(features_dir)

cluster = Cluster(
        n_components=100,
        n_clusters=20,
        random_state=42
)

clusters_df = cluster.create_clusters(features=features_dir, clusters=clusters_dir)
cluster.save_clusters_as_images(clusters=clusters_df, destination=images_dir)
