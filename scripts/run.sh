#!/usr/bin/env bash
set -e

# Full train
python -m src.main --path data/dataset/train --save-matrix data/matrix --all

# Full eval
python -m src.main --load-matrix data/matrix --input-playlists data/dataset/eval/test_input_playlists.json --evaluate data/dataset/eval
