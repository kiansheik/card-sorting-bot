#!/bin/bash

# This function takes a directory containing .jpg images and creates a
# 'train', 'validation', and 'test' directory, then randomly assigns
# 80% of the images to 'train', 10% to 'validation', and the rest to 'test'

function split_data {
  # Check that the input directory exists
  if [ ! -d "$1" ]; then
    echo "Error: input directory does not exist"
    return 1
  fi

  # Create the output directories
  mkdir -p "$1/train" "$1/validation" "$1/test"

  # Get a list of all the .jpg files in the input directory
  files=($1/*.jpg)

  # Calculate the number of images in each set
  num_files=${#files[@]}
  train_size=$((num_files * 8 / 10))
  val_size=$((num_files * 1 / 10))
  test_size=$((num_files - train_size - val_size))

  # Shuffle the list of files
  shuffle() {
    local i tmp size max rand

    size=${#files[*]}
    max=$((32768 / size * size))

    for ((i=size-1; i>0; i--)); do
      while (((rand=$RANDOM) >= max)); do :; done
      rand=$((rand % (i+1)))
      tmp=${files[i]} files[i]=${files[rand]} files[rand]=$tmp
    done
  }
  shuffle

  # Copy the first $train_size images to the 'train' directory
  for ((i=0; i<train_size; i++)); do
    mv "${files[i]}" "$1/train"
  done

  # Copy the next $val_size images to the 'validation' directory
  for ((i=train_size; i<train_size+val_size; i++)); do
    mv "${files[i]}" "$1/validation"
  done

  # Copy the rest of the images to the 'test' directory
  for ((i=train_size+val_size; i<num_files; i++)); do
    mv "${files[i]}" "$1/test"
  done
}

split_data $1