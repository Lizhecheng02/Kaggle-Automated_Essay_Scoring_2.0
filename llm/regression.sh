python regression.py --train_file "../dataset/train.csv" --model_name "google/gemma-2b" --max_length 1536 --num_split 10 --fold_id 0 --lora_r 64 --lora_alpha 16 --lora_dropout 0.1 --warmup_ratio 0.1 --learning_rate 2e-4 --batch_size 8 --accumulation_steps 4 --weight_decay 0.001 --epochs 3 --steps 100 --save_total_limit 10 --lr_scheduler "cosine"