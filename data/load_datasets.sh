#!/bin/bash

# source ./venv/bin/activate venv

# Variables
REPO_URL="https://github.com/Open-Reaction-Database/ord-data"
BRANCH="main"  # Change this to the desired branch if not main
ARCHIVE_NAME="ord-data.zip"
TARGET_DIR="./data/import"

# Step 1: Ensure the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating target directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# Step 2: Check if the target directory is empty
if [ "$(ls -A $TARGET_DIR)" ]; then
    echo "Target directory '$TARGET_DIR' is not empty. Clearing its contents..."
    rm -rf "$TARGET_DIR"/*
else
    echo "Target directory '$TARGET_DIR' is empty."
fi

# Step 3: Download the zip of the current branch
echo "Downloading the $BRANCH branch of the OpenReactionDatabase..."
curl -L -o $ARCHIVE_NAME "$REPO_URL/archive/refs/heads/$BRANCH.zip"

# Step 4: Unzip the archive
echo "Extracting the 'data' folder from the archive..."
unzip -q $ARCHIVE_NAME
EXTRACTED_DIR="ord-data-$BRANCH"

# Check if the extraction was successful
if [ -d "$EXTRACTED_DIR/data" ]; then
    echo "'data' folder found in the archive."
else
    echo "Error: 'data' folder not found in the archive. Please check the structure of the downloaded zip file."
    rm -rf "$EXTRACTED_DIR"
    rm $ARCHIVE_NAME
    exit 1
fi

# Step 5: Move the 'data' folder to the target directory
echo "Moving 'data' folder to $TARGET_DIR..."
mv "$EXTRACTED_DIR/data/"* "$TARGET_DIR"

# Step 6: Clean up
echo "Cleaning up temporary files..."
rm -rf $EXTRACTED_DIR
rm $ARCHIVE_NAME

python3 ./data/extract.py

echo "Script successfully finished!"

# deactivate venv