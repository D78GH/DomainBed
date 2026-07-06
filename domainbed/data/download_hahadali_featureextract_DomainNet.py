import os
import glob
import h5py
import numpy as np
import torch

from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./domain_net"
OUT_DIR = "./DomainNetFeatures"

DOMAINS = [
    "sketch",
    "real",
    "quickdraw",
    "painting",
    "infograph",
    "clipart"
]

VAL_SPLIT = {
    "sketch": 0.10,
    "real": 0.10,
    "quickdraw": 0.10,
    "painting": 0.10,
    "infograph": 0.10,
    "clipart": 0.10
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL
# ============================================================

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_paths, labels):

    feats, labs = [], []

    with torch.no_grad():
        for i, (p, y) in enumerate(zip(image_paths, labels)):

            img = Image.open(p).convert("RGB")

            x = transform(img).unsqueeze(0).to(DEVICE)

            f = feature_extractor(x).flatten(1).cpu().numpy()[0]

            feats.append(f)
            labs.append(y)

            if (i + 1) % 500 == 0:
                print(f"Processed {i+1}/{len(image_paths)}")

    return (
        np.array(feats, dtype=np.float32),
        np.array(labs, dtype=np.int64)
    )


# ============================================================
# LOAD VLCS FROM DIRECTORY ONLY
# ============================================================
def load_vlcs_from_folder(data_dir):

    import os
    import glob

    domains = [
        "sketch",
        "real",
        "quickdraw",
        "painting",
        "infograph",
        "clipart"
    ]

    data = {}

    for d in domains:

        domain_path = os.path.join(data_dir, d)

        class_names = sorted([
            c for c in os.listdir(domain_path)
            if os.path.isdir(os.path.join(domain_path, c))
        ])

        class_to_idx = {c: i for i, c in enumerate(class_names)}

        images = []
        labels = []

        for cls in class_names:

            cls_path = os.path.join(domain_path, cls)

            for ext in ["jpg", "jpeg", "png", "JPG", "JPEG"]:
                files = glob.glob(os.path.join(cls_path, f"*.{ext}"))

                images.extend(files)
                labels.extend([class_to_idx[cls]] * len(files))

        print(f"{d}: {len(images)} images, {len(class_names)} classes")

        data[d] = (images, labels)

    return data

# ============================================================
# SAVE HDF5
# ============================================================

def save_hdf5(path, x, y):
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=x, compression="gzip")
        f.create_dataset("labels", data=y, compression="gzip")

    print("Saved:", path)


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_hdf5():

    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_vlcs_from_folder(DATA_DIR)

    for domain in DOMAINS:

        print("\n==============================")
        print("DOMAIN:", domain)
        print("==============================")

        images, labels = data[domain]

        train_x, val_x, train_y, val_y = train_test_split(
            images,
            labels,
            test_size=VAL_SPLIT[domain],
            random_state=0,
            shuffle=True
        )

        print("Extracting train...")
        tr_f, tr_l = extract_features(train_x, train_y)

        print("Extracting val...")
        va_f, va_l = extract_features(val_x, val_y)

        features_f = np.concatenate([tr_f, va_f], axis=0)
        features_l = np.concatenate([tr_l, va_l], axis=0)

        save_hdf5(os.path.join(OUT_DIR, f"{domain}_train.hdf5"), tr_f, tr_l)
        save_hdf5(os.path.join(OUT_DIR, f"{domain}_val.hdf5"), va_f, va_l)
        save_hdf5(os.path.join(OUT_DIR, f"{domain}_features.hdf5"), features_f, features_l)

    print("\nDONE")


if __name__ == "__main__":
    build_hdf5()