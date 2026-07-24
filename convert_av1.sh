#!/bin/bash
set -euo pipefail

# Pointing to YOUR copied directories
GREEN_DIR="/iris/projects/humanoid/trossen_data/0528_green_block_mem_copy/videos/chunk-000/observation.images.cam_high"
YELLOW_DIR="/iris/projects/humanoid/trossen_data/0528_yellow_block_mem_copy/videos/chunk-000/observation.images.cam_high"

DIRECTORIES=("$GREEN_DIR" "$YELLOW_DIR")

for TARGET_DIR in "${DIRECTORIES[@]}"; do
    echo "========================================"
    echo "Processing directory: $TARGET_DIR"
    
    OUT_DIR="$TARGET_DIR/h264_videos"
    mkdir -p "$OUT_DIR"

    for video in "$TARGET_DIR"/*.mp4; do
        [ -e "$video" ] || continue 
        
        filename=$(basename "$video")
        echo "Converting $filename..."
        
        ffmpeg -y -i "$video" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "$OUT_DIR/$filename" -loglevel warning
    done
done

echo "========================================"
echo "✅ All videos successfully converted to H.264!"