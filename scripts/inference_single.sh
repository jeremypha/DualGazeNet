
BACKBONE=${1:-"L"}
CHECKPOINT=${2:-"path/to/your/checkpoint"}
IM_DIR=${3:-"path/to/your/images"}                      

echo "Starting DualGazeNet inference..."
echo "Task: $TASK"
echo "Backbone: $BACKBONE"
echo "Checkpoint: $CHECKPOINT"

python inference_single.py \
    --backbone "$BACKBONE" \
    --checkpoint "$CHECKPOINT" \
    --im_dir "$IM_DIR" \
    --pred_dir './pred/single' \
    --input_size 512 \
    --image_ext ".jpg" \
    --device "cuda"