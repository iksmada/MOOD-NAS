#!/bin/bash

# ===============================================================================================================
# Make subsample selection of images from ImageNet (ILSVRC2012)
# or any other folder with subdirectories for each class, like:
#        root/dog/xxx.png
#        root/dog/xxy.png
#        root/cat/123.png
#        root/cat/nsdf3.png
#
# Usage: subsample.sh <root_dir> <dest_dir>
#
# The root dir should contain one folder per class. the folders can be compacted using .tar format
# The destination dir will have the following structure:
#        dest/imagenet_search/[train|val]/class/picture.png
# The subsample per class for training is 10% (130) while validation has 2,5% (32)
# ===============================================================================================================

source_dir=$1
output_dir=$2

i=0
find "$source_dir" -maxdepth 1 -type f  -name "*.tar" | while read -r tar
do
  ((i++))
  dir="${tar%.*}"
  echo "Untar class $(basename "$dir") $i/1000"
  mkdir "$dir";
  tar -xf "$tar" -C "$dir" && rm "$tar"
done

train_dest="${output_dir}/imagenet_search/train"
valid_dest="${output_dir}/imagenet_search/val"
i=0
for class in "${source_dir}/"*/
do
  ((i++))
  echo "Subsampling class ${class} $i/1000"
  files=($(find "$class" -type f))
  total="${#files[@]}"
  subsample_train=$((total / 10)) #10 %
  subsample_val=$((total * 25 / 1000 )) #2.5%

  echo "Sampling $subsample_train images to train"
  for j in $(seq 1 $subsample_train)
  do
    image_index=$(( $RANDOM % "${#files[@]}" ))
    file=${files[ $image_index ]}
    #echo "Drew $file"
    dest="$train_dest/$(basename "$class")/"
    mkdir -p "$dest"
    cp "$file" "$dest"
    unset 'files[image_index]'
    files=("${files[@]}")
  done


  echo "Sampling $subsample_val images to validation"
  for j in $(seq 1 $subsample_val)
  do
    image_index=$(( $RANDOM % "${#files[@]}" ))
    file=${files[ $image_index ]}
    #echo "Drew $file"
    dest="$valid_dest/$(basename "$class")/"
    mkdir -p "$dest"
    cp "$file" "$dest"
    unset 'files[image_index]'
    files=("${files[@]}")
  done
done

train_log="train.log"
rm -rf $train_log
valid_log="valid.log"
rm -rf $valid_log
echo "Sanity checks"
train=$(find "$train_dest" -type f | wc -l)
for folder in "$train_dest/"*
do
  find "$folder" -type f | wc -l >> $train_log
done
echo " Total train files: $train"
echo " Total folders: $( wc -l $train_log ), where $(sort $train_log  |uniq -c | grep "130" | awk '{print $1;}') have 130 images"
valid=$(find "$valid_dest" -type f | wc -l)
for folder in "$valid_dest/"*
do
  find "$folder" -type f | wc -l >> $valid_log
done
echo " Total validation files: $valid"
echo " Total folders: $( wc -l $valid_log ), where $(sort $valid_log  | uniq -c | grep "32" | awk '{print $1;}') have 32 images"
