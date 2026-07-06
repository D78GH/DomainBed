import h5py
import numpy as np
import torch

from datasets import load_dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DOMAINS = [
    "art_painting",
    "cartoon",
    "photo",
    "sketch"
]

# Match the old HDF5 generation behaviour:
#
# art_painting:
#   80% train / 20% val
#
# others:
#   90% train / 10% val
#
VAL_SPLIT = {
    "art_painting": 0.20,
    "cartoon": 0.10,
    "photo": 0.10,
    "sketch": 0.10
}

RANDOM_SEED = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD PACS
# ============================================================

print("Loading PACS dataset...")

ds = load_dataset("flwrlabs/pacs")
dataset = ds["train"]

print(
    f"Total PACS samples: {len(dataset)}"
)


# ============================================================
# RESNET18 FEATURE EXTRACTOR
# ============================================================

print("Loading pretrained ResNet18...")


resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor.eval()


# ============================================================
# TRANSFORM
# ============================================================

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

def extract_features(samples):

    features = []
    labels = []

    with torch.no_grad():

        for idx, sample in enumerate(samples):

            image = sample["image"].convert("RGB")

            x = transform(image)
            x = x.unsqueeze(0).to(DEVICE)

            feat = feature_extractor(x)

            # (1,512,1,1) -> (512,)
            feat = feat.flatten(1)
            feat = feat.cpu().numpy()[0]

            features.append(feat)
            labels.append(sample["label"])

            if (idx + 1) % 500 == 0:
                print(
                    f"  Processed {idx+1}/{len(samples)}"
                )


    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64)
    )


# ============================================================
# HDF5 WRITER
#
# keys:
#   images -> (N,512)
#   label  -> (N,)
# ============================================================

def save_hdf5(filename, features, labels):

    with h5py.File(filename, "w") as f:

        f.create_dataset(
            "images",
            data=features,
            compression="gzip"
        )

        f.create_dataset(
            "label",
            data=labels,
            compression="gzip"
        )

    print(
        f"Saved {filename}: "
        f"images={features.shape}, "
        f"label={labels.shape}"
    )


# ============================================================
# GENERATE TRAIN / VAL / FULL
# ============================================================

for domain in DOMAINS:

    print("\n" + "=" * 60)
    print(f"PROCESSING DOMAIN: {domain}")
    print("=" * 60)


    domain_samples = [
        sample
        for sample in dataset
        if sample["domain"] == domain
    ]

    print(
        f"Domain samples: {len(domain_samples)}"
    )


    labels_for_split = [
        sample["label"]
        for sample in domain_samples
    ]


    split = VAL_SPLIT[domain]

    print(
        f"Using validation split: {split}"
    )


    train_samples, val_samples = train_test_split(
        domain_samples,
        test_size=split,
        random_state=RANDOM_SEED,
        stratify=labels_for_split,
        shuffle=True
    )


    print(
        f"Train: {len(train_samples)} | "
        f"Val: {len(val_samples)}"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("Extracting TRAIN features...")

    train_features, train_labels = extract_features(
        train_samples
    )

    save_hdf5(
        f"{domain}_train_features.hdf5",
        train_features,
        train_labels
    )


    # --------------------------------------------------------
    # VAL
    # --------------------------------------------------------

    print("Extracting VAL features...")

    val_features, val_labels = extract_features(
        val_samples
    )

    save_hdf5(
        f"{domain}_val_features.hdf5",
        val_features,
        val_labels
    )


    # --------------------------------------------------------
    # FULL
    #
    # Same order as train + val
    # --------------------------------------------------------

    full_features = np.concatenate(
        [
            train_features,
            val_features
        ],
        axis=0
    )

    full_labels = np.concatenate(
        [
            train_labels,
            val_labels
        ],
        axis=0
    )


    save_hdf5(
        f"{domain}_features.hdf5",
        full_features,
        full_labels
    )


print("Done.")