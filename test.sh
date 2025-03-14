EXP_DIR=checkpoints/
CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch\
    --nproc_per_node 2 \
    --master_port 3306 \
    test.py  \
    --dataset rrsisd \
    --swin_type base \
    --resume ${EXP_DIR}/model_best_RMSIN.pth \
    --batch-size 8 \
    --split test\
    --workers 4 \
    --window12 \
    --model_id RMSIN \
    --epochs 40 \
    --img_size 480\
    --kernels 1 3 5\
    --output_dir ${EXP_DIR} \
    2>&1 | tee ./output