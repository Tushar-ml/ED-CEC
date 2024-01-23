
PRETRAINED_MODEL=bert-base-uncased
python -u ../train.py \
    --base_model $PRETRAINED_MODEL \
    --tokenizer_name $PRETRAINED_MODEL \
    --train_data_file ../datasets/atis/valid.json \
    --eval_data_file ../datasets/atis/valid.json \
    --tag_pdrop 0.2 \
    --decoder_proj_pdrop 0.2 \
    --tag_hidden_size 768 \
    --tag_size 3 \
    --device mps \
    --vocab_size 30522 \
    --pad_token_id 0 \
    --alpha 3.0 \
    --change_weight 1.5 \
    --max_src_len 256 \
    --max_add_len 10 \
    --batch_size 32 \
    --lr 5e-5 \
    --max_num_epochs 1 \
    --rare_words_list ../datasets/atis/valid_rare.txt \
    --save_dir ./models.atis/