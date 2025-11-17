TASK=${1:-"SOD"}
BACKBONE=${2:-"L"}

python train.py \
    --mode train \
    --task $TASK \
    --backbone $BACKBONE \
    --input_size 512 \
    --device cuda \
    --batch_size_train 8 \
    --max_epoch_num 100 \
    --model_save_fre 3 \
    --eval_interval 3 \
    --output_dir "./output" \
    --resume_cpt "" \