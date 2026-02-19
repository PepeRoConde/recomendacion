#!/usr/bin/env bash
set -e

# Quick train (5 slices)
python -m src.main --path data/dataset/train --quick --save-matrix data/matrix --all

# Quick eval (200 playlists, sampled)
python -m src.main --load-matrix data/matrix --input-playlists data/dataset/eval/test_input_playlists.json --evaluate data/dataset/eval --eval-sample 200
