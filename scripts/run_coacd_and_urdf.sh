#!/usr/bin/env bash
# Run CoACD on all object meshes in data/OakInk-v2/object_preview/align_ds,
# then generate URDF files for each result in data/OakInk-v2/coacd_object_preview/align_ds.
#
# Prerequisites:
#   pip install trimesh coacd
#
# Usage (from repo root):
#   bash scripts/run_coacd_and_urdf.sh [--jobs N]
#
# Options:
#   --jobs N   Number of parallel workers for CoACD (default: 4)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="data/OakInk-v2/object_preview/align_ds"
OUTPUT_DIR="data/OakInk-v2/coacd_object_preview/align_ds"
JOBS=4

while [[ $# -gt 0 ]]; do
  case $1 in
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure object_preview exists (symlink to object_repair if needed)
OAK_DIR="data/OakInk-v2"
if [[ ! -d "$OAK_DIR/object_preview" && -d "$OAK_DIR/object_repair" ]]; then
  echo "Creating object_preview -> object_repair symlink..."
  ln -snf object_repair "$OAK_DIR/object_preview"
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input dir not found: $INPUT_DIR"
  echo "Extract object_repair (e.g. tar -xf object_repair.tar) and ensure object_preview/align_ds exists."
  exit 1
fi

echo "Step 1: CoACD decomposition (input=$INPUT_DIR, output=$OUTPUT_DIR, jobs=$JOBS)"
python scripts/batch_coacd.py \
  --input_dir  "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --jobs       "$JOBS"

echo ""
echo "Step 2: Generate URDF files for each CoACD output"
python scripts/gen_urdf.py --coacd_dir "$OUTPUT_DIR"

echo ""
echo "Optional: copy obj_desc.json to coacd_object_preview for downstream use"
if [[ -f "$OAK_DIR/object_repair/obj_desc.json" && ! -f "$OAK_DIR/coacd_object_preview/obj_desc.json" ]]; then
  mkdir -p "$OAK_DIR/coacd_object_preview"
  cp "$OAK_DIR/object_repair/obj_desc.json" "$OAK_DIR/coacd_object_preview/obj_desc.json"
  echo "  Copied obj_desc.json"
else
  echo "  Skipped (already present or source missing)"
fi

echo ""
echo "Done."
