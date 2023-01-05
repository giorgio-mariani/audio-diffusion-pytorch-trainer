#!/bin/env bash
STEM='other' # TODO: put desired stem
MUSDB_DIR='/data/MusDB/data/musdb18hq' # TODO: put path to dataset

# The following instructions extract the stem from the dataset and put it in a folder
# with the stem name. The folder contains the test and train folders.

mkdir -p "$STEM"

for SPLIT_DIR in "$MUSDB_DIR"/*;
do
	SPLIT_DIR="$(basename $SPLIT_DIR)"
	echo ""
	echo "========================================="
	echo $SPLIT_DIR
	echo "========================================="
	mkdir -p "$STEM/$SPLIT_DIR"
	for TRACK_DIR in "$MUSDB_DIR/$SPLIT_DIR"/*;
	do 
		TRACK_DIR=$(basename "${TRACK_DIR}")
		echo $TRACK_DIR
		cp "$MUSDB_DIR/$SPLIT_DIR/$TRACK_DIR/$STEM.wav" "$STEM/$SPLIT_DIR/${TRACK_DIR}.wav"
	done
done
