#!/usr/bin/env bash
set -euo pipefail

dataset_path="data/sift-128-euclidean.hdf5"
dataset_url="https://ann-benchmarks.com/sift-128-euclidean.hdf5"

mkdir -p data

if [[ -f "$dataset_path" ]]; then
  echo "Dataset already exists. Skipping download."
  exit 0
fi

echo "Downloading SIFT1M dataset..."

if command -v curl >/dev/null 2>&1; then
  curl -L -o "$dataset_path" "$dataset_url"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$dataset_path" "$dataset_url"
else
  echo "Error: neither curl nor wget is installed."
  exit 1
fi

echo "Download complete."
