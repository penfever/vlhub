#!/bin/bash

# Directory to search
dir="/scratch/bf996/vlhub/JANuS_runs"

# Use find to recursively search for files matching the pattern
find "$dir" -type f -name "epoch_*.pt" | while read file
do
    # Extract epoch number
    epoch_num=$(basename "$file" | sed -n 's/^epoch_\([0-9]\+\).pt$/\1/p')

    # If the file name does not match the pattern, skip
    if [ -z "$epoch_num" ]; then
        continue
    fi

    # If epoch number is less than 200, remove the file
    if [ "$epoch_num" -lt 224 ]; then
        if [ "$epoch_num" -gt 32 ]; then
            echo "Removing $file"
            rm "$file"
        fi
    fi
done