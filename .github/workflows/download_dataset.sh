#!/bin/bash

dataset_dir=$DIR/dataset
cache_tar_dir=$DIR/../dataset

[ ! -d $cache_tar_dir ] && mkdir $cache_tar_dir

wget="wget -q"

# download imagenet val set
img_val_tar=ILSVRC2012_img_val.tar
img_val_tar_path=$cache_tar_dir/$img_val_tar
[ ! -s $img_val_tar_path ] && $wget -O \
    $img_val_tar_path \
    $IMG_VAL_URL > /dev/null
tar -xf $img_val_tar_path -C $dataset_dir/ILSVRC2012/ILSVRC2012_img_val

# download MS COCO annotations
[ -z $COCO_ANNO_URL ] && \
COCO_ANNO_URL=http://images.cocodataset.org/annotations/annotations_trainval2017.zip
coco_anno_zip=annotations_trainval2017.zip
coco_anno_zip_path=$cache_tar_dir/$coco_anno_zip
[ ! -s $coco_anno_zip_path ] && $wget -O \
    $coco_anno_zip_path \
    $COCO_ANNO_URL > /dev/null
unzip -o -d $dataset_dir/COCO2017 $coco_anno_zip_path > /dev/null

# download MS COCO val set
[ -z $COCO_VAL_URL ] && \
COCO_VAL_URL=http://images.cocodataset.org/zips/val2017.zip
coco_val_zip=val2017.zip
coco_val_zip_path=$cache_tar_dir/$coco_val_zip
[ ! -s $coco_val_zip_path ] && $wget -O \
    $coco_val_zip_path \
    $COCO_VAL_URL > /dev/null
unzip -o -d $dataset_dir/COCO2017 $coco_val_zip_path > /dev/null