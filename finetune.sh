# SIZE=small
SIZE=large
DATA_PATH=esc
# DATA_PATH=data
SAVE_DIR=test_${SIZE}_${DATA_PATH}


env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node=8 \
    code/finetune_trainer.py \
    --data_dir=$DATA_PATH \
    --train_name=train \
    --max_target_length 128 \
    --max_source_length 128 \
    --val_max_target_length 128 \
    --test_max_target_length 128 \
    --output_dir=results/$SAVE_DIR \
    --save_total_limit=10 \
    --per_gpu_train_batch_size=3 \
    --per_gpu_eval_batch_size=3 \
    --num_train_epochs=1 \
    --logging_steps=1 \
    --model_name_or_path=LongLM-$SIZE \
    --warmup_steps=100 \
    --learning_rate=1e-4 \
    --n_val=100 \
    --do_train --do_eval \
    --evaluation_strategy steps \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --gradient_accumulation_steps=40