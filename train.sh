EXP_DIR=exps/xiaorong/mqn2
CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch\
    --nproc_per_node 2 \
    --master_port 3306 \
    train.py  \
    --dataset rrsisd \
    --batch-size 8\
    --model_id RMSIN \
    --epochs 40 \
    --img_size 480\
    --kernels 1 3 5\
    --debug \
    --output_dir ${EXP_DIR} \
    2>&1 | tee ./output

    ##backbone1 gemimi
    ##backbone3 corr
    #backbone4  corr+gemimi
    ##backbone5  clip text_encode