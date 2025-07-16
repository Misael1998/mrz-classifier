import os
import torch
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import imagehash

class ISSDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_paths = []
        self.transform = transform
        seen_hashes = set()

        for fname in os.listdir(img_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(img_dir, fname)
                try:
                    img = Image.open(path).convert('RGB')
                    h = str(imagehash.phash(img))
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        self.img_paths.append(path)
                except:
                    continue

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, path

class Model():
    def __init__(self):
        self.all_features = []
        self.all_paths = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

        resnet50 = models.resnet50(weights=True) 
        resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
        resnet50.to(self.device)
        resnet50.eval()

        self.model = resnet50

    def extract_features(self, dir):
        dataset = ISSDataset(dir, self.transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)


        with torch.no_grad():
            for imgs, paths in tqdm(loader):
                imgs = imgs.to(self.device)
                feats = self.model(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
                self.all_features.extend(feats)
                self.all_paths.extend(paths)

    def save_features(self, path = 'features_50.csv'):
        features = np.array(self.all_features)
        df = pd.DataFrame(features)
        df['path'] = self.all_paths
        df.to_csv(path, index=False)

