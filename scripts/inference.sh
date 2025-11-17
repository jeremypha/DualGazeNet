TASK=${1:-"SOD"}
BACKBONE=${2:-"L"}
CHECKPOINT=${3:-""}


echo "Starting DualGazeNet inference..."
echo "Task: $TASK"
echo "Backbone: $BACKBONE"
echo "Checkpoint: $CHECKPOINT"

python inference.py \
    --mode valid \
    --task $TASK \
    --backbone $BACKBONE \
    --input_size 352 \
    --device cuda \
    --resume_cpt "$CHECKPOINT" \
    --visualize True \
    --pred_dir "./pred"