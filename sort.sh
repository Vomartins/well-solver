#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
  echo "Please provide the input file."
  exit 1
fi

# Save the sorted output to a temporary file, then overwrite the original file
sort -s -t ',' -k1.7n "$1" > temp_sorted.txt && mv temp_sorted.txt "$1"
