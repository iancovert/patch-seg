#!/bin/bash

# Prepare directories
mkdir coco
cd coco
mkdir images
cd images

# Download images
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip
wget -c http://images.cocodataset.org/zips/unlabeled2017.zip

# Uncompress images
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip unlabeled2017.zip

# Remove image zip files
rm train2017.zip
rm val2017.zip
rm test2017.zip
rm unlabeled2017.zip

# Download annotations
cd ../
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

# Uncompress annotations
unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
unzip image_info_test2017.zip
unzip image_info_unlabeled2017.zip

# Remove annotation zip files
rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
rm image_info_unlabeled2017.zip

# Coco-stuff annotations
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
unzip stuffthingmaps_trainval2017.zip -d annotations/
rm stuffthingmaps_trainval2017.zip
