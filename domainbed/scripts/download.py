# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import defaultdict
# from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import gdown
import uuid
import json
import os
import urllib
import zipfile
import os
import urllib.request
import zipfile

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset
from datasets import load_dataset
from tqdm import tqdm

# utils #######################################################################

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


# VLCS ########################################################################
# Slower, but builds dataset from the original sources
def download_vlcs(data_dir):
    full_path = stage_path(data_dir, "VLCS")
    tmp_path = os.path.join(full_path, "tmp")
    os.makedirs(tmp_path, exist_ok=True)

    # load mapping file
    with open("domainbed/misc/vlcs_files.txt", "r") as f:
        files = [line.strip().split() for line in f]

    # VOC2007 (PASCAL VOC)
    voc_tar = os.path.join(tmp_path, "voc2007.tar")
    download_and_extract(
        "http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
        voc_tar
    )

    voc_root = os.path.join(tmp_path, "VOCdevkit", "VOC2007", "JPEGImages")

    # CALTECH
    caltech_zip = os.path.join(tmp_path, "caltech101.zip")
    download_and_extract(
        "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1",
        caltech_zip
    )

    # extract nested tar for CALTECH
    nested_tar = None
    for root, _, files_ in os.walk(tmp_path):
        if "101_ObjectCategories.tar.gz" in files_:
            nested_tar = os.path.join(root, "101_ObjectCategories.tar.gz")
            break

    if nested_tar is None:
        raise RuntimeError("Caltech nested tar not found")

    with tarfile.open(nested_tar, "r:gz") as tar:
        tar.extractall(os.path.dirname(nested_tar))

    caltech_root = None
    for root, dirs, _ in os.walk(tmp_path):
        if root.endswith("101_ObjectCategories"):
            caltech_root = root
            break

    if caltech_root is None:
        raise RuntimeError("Caltech root not found")

    # SUN09
    sun_tar = os.path.join(tmp_path, "sun09_hcontext.tar")
    download_and_extract(
        "https://groups.csail.mit.edu/vision/Hcontext/data/sun09_hcontext.tar",
        sun_tar
    )

    # copy files to VLCS structure
    for src, dst in files:

        print(f"\nProcessing: {src}")

        if dst.startswith("VLCS/"):
            dst = dst[len("VLCS/"):]

        class_folder = os.path.join(full_path, dst)
        os.makedirs(class_folder, exist_ok=True)

        dst_file = os.path.join(class_folder, uuid.uuid4().hex + ".jpg")

        # LABELME
        if "labelme" in src:
            gdown.download(src, dst_file, quiet=False)
            continue

        # CALTECH
        if src.startswith("101_ObjectCategories/"):
            src_path = os.path.join(
                caltech_root,
                src.replace("101_ObjectCategories/", "")
            )

        # OTHER DATASETS (VOC + SUN09)
        else:
            src_path = os.path.join(tmp_path, src)

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing file: {src_path}")

        shutil.copyfile(src_path, dst_file)
        
    # Clean up temporary download/extraction directory
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

# MNIST #######################################################################

def download_mnist(data_dir):
    # Original URL: http://yann.lecun.com/exdb/mnist/
    full_path = stage_path(data_dir, "MNIST")
    MNIST(full_path, download=True)


# PACS ########################################################################

def download_pacs(data_dir):

    full_path = os.path.join(data_dir, "PACS")
    os.makedirs(full_path, exist_ok=True)

    print("Loading PACS dataset...")
    train_ds = load_dataset("flwrlabs/pacs", data_files="/home/s2457428/DomainBed/domainbed/data/PACS/",  split="train")

    print(f"Dataset size: {len(train_ds)}")

    label_names = train_ds.features["label"].names
    domain_names = sorted(set(train_ds["domain"]))

    print("Domains:", domain_names)
    print("Classes:", label_names)

    # Create directory structure
    for domain in domain_names:
        for label in label_names:
            os.makedirs(
                os.path.join(full_path, domain, label),
                exist_ok=True
            )

    print("Exporting images...")

    for idx, sample in enumerate(tqdm(train_ds)):
        image = sample["image"]
        domain = sample["domain"]
        label_name = label_names[sample["label"]]

        filename = os.path.join(
            full_path,
            domain,
            label_name,
            f"{idx}.jpg"
        )

        image.save(filename)

    print(f"\nPACS exported successfully to:")
    print(full_path)

    total_files = sum(
        len(files)
        for _, _, files in os.walk(full_path)
    )

    print(f"Saved {total_files} images.")

# Office-Home #################################################################

# def download_office_home(data_dir):
#     # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
#     full_path = stage_path(data_dir, "office_home")

#     download_and_extract("https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC",
#                          os.path.join(data_dir, "office_home.zip"))

#     os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
#               full_path)

def download_office_home(data_dir):

    full_path = os.path.join(data_dir, "OfficeHome")
    os.makedirs(full_path, exist_ok=True)

    print("Loading office-home dataset...")
    train_ds = load_dataset("flwrlabs/office-home", split="train")

    print(f"Dataset size: {len(train_ds)}")

    label_names = train_ds.features["label"].names
    domain_names = sorted(set(train_ds["domain"]))

    print("Domains:", domain_names)
    print("Classes:", label_names)

    # Create directory structure
    for domain in domain_names:
        for label in label_names:
            os.makedirs(
                os.path.join(full_path, domain, label),
                exist_ok=True
            )

    print("Exporting images...")

    for idx, sample in enumerate(tqdm(train_ds)):
        image = sample["image"]
        domain = sample["domain"]
        label_name = label_names[sample["label"]]

        filename = os.path.join(
            full_path,
            domain,
            label_name,
            f"{idx}.jpg"
        )

        image.save(filename)

    print(f"\noffice-home exported successfully to:")
    print(full_path)

    total_files = sum(
        len(files)
        for _, _, files in os.walk(full_path)
    )

    print(f"Saved {total_files} images.")

# DomainNET ###################################################################

def download_domain_net(data_dir):
    # Original URL: http://ai.bu.edu/M3SDA/
    full_path = stage_path(data_dir, "domain_net")

    urls = [
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
    ]

    for url in urls:
        download_and_extract(url, os.path.join(full_path, url.split("/")[-1]))

    with open("domainbed/misc/domain_net_duplicates.txt", "r") as f:
        for line in f.readlines():
            try:
                os.remove(os.path.join(full_path, line.strip()))
            except OSError:
                pass


# TerraIncognita ##############################################################

def download_terra_incognita(data_dir):
    # Original URL: https://beerys.github.io/CaltechCameraTraps/
    # New URL: http://lila.science/datasets/caltech-camera-traps

    full_path = stage_path(data_dir, "terra_incognita")

    download_and_extract(
        "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz",
        os.path.join(full_path, "terra_incognita_images.tar.gz"))


    download_and_extract(
        "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz",
        os.path.join(full_path, "eccv_18_annotations.tar.gz"))


    include_locations = ["38", "46", "100", "43"]

    include_categories = [
        "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
        "raccoon", "squirrel"
    ]

    images_folder = os.path.join(full_path, "eccv_18_all_images_sm/")
    annotations_folder = os.path.join(full_path,"eccv_18_annotation_files/")
    cis_test_annotations_file = os.path.join(full_path, "eccv_18_annotation_files/cis_test_annotations.json")
    cis_val_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/cis_val_annotations.json")
    train_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/train_annotations.json")
    trans_test_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/trans_test_annotations.json")
    trans_val_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/trans_val_annotations.json")
    annotations_file_list = [cis_test_annotations_file, cis_val_annotations_file, train_annotations_file, trans_test_annotations_file, trans_val_annotations_file]
    destination_folder = full_path

    stats = {}
    data = defaultdict(list)

    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    for annotations_file in annotations_file_list:
        annots = {}
        with open(annotations_file, "r") as f:
            annots = json.load(f)
            for k, v in annots.items():
                data[k].extend(v)



    category_dict = {}
    for item in data['categories']:
        category_dict[item['id']] = item['name']

    for image in data['images']:
        image_location = str(image['location'])

        if image_location not in include_locations:
            continue

        loc_folder = os.path.join(destination_folder,
                                  'location_' + str(image_location) + '/')

        if not os.path.exists(loc_folder):
            os.mkdir(loc_folder)

        image_id = image['id']
        image_fname = image['file_name']

        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                if image_location not in stats:
                    stats[image_location] = {}

                category = category_dict[annotation['category_id']]

                if category not in include_categories:
                    continue

                if category not in stats[image_location]:
                    stats[image_location][category] = 0
                else:
                    stats[image_location][category] += 1

                loc_cat_folder = os.path.join(loc_folder, category + '/')

                if not os.path.exists(loc_cat_folder):
                    os.mkdir(loc_cat_folder)

                dst_path = os.path.join(loc_cat_folder, image_fname)
                src_path = os.path.join(images_folder, image_fname)

                shutil.copyfile(src_path, dst_path)

    shutil.rmtree(images_folder)
    shutil.rmtree(annotations_folder)



# SVIRO #################################################################

def download_sviro(data_dir):
    # Original URL: https://sviro.kl.dfki.de
    full_path = stage_path(data_dir, "sviro")

    download_and_extract("https://sviro.kl.dfki.de/?wpdmdl=1731",
                         os.path.join(data_dir, "sviro_grayscale_rectangle_classification.zip"))

    os.rename(os.path.join(data_dir, "SVIRO_DOMAINBED"),
              full_path)


# SPAWRIOUS #############################################################

def download_spawrious(data_dir, remove=True):
    dst = os.path.join(data_dir, "spawrious.tar.gz")
    urllib.request.urlretrieve('https://www.dropbox.com/s/e40j553480h3f3s/spawrious224.tar.gz?dl=1', dst)
    tar = tarfile.open(dst, "r:gz")
    tar.extractall(os.path.dirname(dst))
    tar.close()
    if remove:
        os.remove(dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # download_mnist(args.data_dir)
    # download_pacs(args.data_dir)
    # download_office_home(args.data_dir)
    download_domain_net(args.data_dir)
    # download_vlcs(args.data_dir)
    # download_terra_incognita(args.data_dir)
    # download_spawrious(args.data_dir)
    # download_sviro(args.data_dir)
    # Camelyon17Dataset(root_dir=args.data_dir, download=True)
    # FMoWDataset(root_dir=args.data_dir, download=True)
