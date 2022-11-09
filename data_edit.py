import pandas as pd
import os
from PIL import Image

def transform_data(annotations, root_dir, new_root_dir, type, size=(30, 30)):
  dct = {
    "label" : [],
    "path" : [],
  }
  
  annotations = pd.read_csv(annotations)

  for idx in range(len(annotations)):
    original_path = os.path.join(root_dir, annotations.iloc[idx]["Path"])

    final_path = os.path.join(new_root_dir, type)
    final_path = os.path.join(final_path, f"{type}-{idx}.png")

    label = annotations.iloc[idx]["ClassId"]

    dct["path"].append(final_path)
    dct["label"].append(label)
    
    img = Image.open(original_path)
    resized_img = img.resize(size)
    resized_img.save(final_path)

  pd.DataFrame(dct).to_csv(f"{new_root_dir}/{type}-annotations.csv", index=False)

