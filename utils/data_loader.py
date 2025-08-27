import os
import pandas as pd
from pathlib import Path
import keras

class DataLoader():
    def __init__(self, path):
        self.path = path

    def get_data(self, img_size=(255,255)):
        labels = [name for name in os.listdir(self.path)]
        data = []

        for label in labels:
            label_data_path = f'{self.path}/{label}'
            lbl_pth = Path(label_data_path)
            
            for img_path in lbl_pth.iterdir():
                image = keras.preprocessing.image.load_img(img_path, target_size=img_size)
                img_array = keras.preprocessing.image.img_to_array(image)
                img_array = img_array / 255

                data.append({
                    "label": label,
                    "data": img_array
                    })

        return pd.DataFrame(data)


