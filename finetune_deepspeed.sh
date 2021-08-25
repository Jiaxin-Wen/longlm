# SIZE=small
SIZE=large
DATA_PATH=esc
# DATA_PATH=data
SAVE_DIR=test_${SIZE}_${DATA_PATH}_deepspeed


export NCCL_DEBUG=INFO
deepspeed  \
    code/finetune_trainer.py \
    --data_dir=$DATA_PATH \
    --train_name=train \
    --max_target_length 128 \
    --max_source_length 128 \
    --val_max_target_length 128 \
    --test_max_target_length 128 \
    --output_dir=results/$SAVE_DIR \
    --save_total_limit=10 \
    --per_device_train_batch_size=3 \
    --per_device_eval_batch_size=3 \
    --num_train_epochs=1 \
    --logging_steps=5 \
    --model_name_or_path=LongLM-$SIZE \
    --learning_rate=1e-4 \
    --n_val=100 \
    --evaluation_strategy=steps \
    --eval_steps=100 \
    --do_train --do_eval \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --gradient_accumulation_steps 40 \
    --deepspeed config/ds_zero2_config_st.json
