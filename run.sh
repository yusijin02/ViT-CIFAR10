cd src
function run() {
    python main.py \
        --epoch=$EPOCH \
        --batch_size=$BATCH_SIZE \
        --embedding_dim=$EMBEDDING_DIM \
        --picture_size=$PICTURE_SIZE \
        --num_classes=$NUM_CLASSES \
        --patch_size=$PATCH_SIZE \
        --hidden_size=$HIDDEN_SIZE \
        --mlp_hidden_size=$MLP_HIDDEN_SIZE \
        --num_layer=$NUM_LAYER \
        --num_heads=$NUM_HEADS \
        --steplr_step_size=$STEPLR_STEP_SIZE
}


EPOCH=250             # epoch数量
BATCH_SIZE=256        # batch大小  # 64 acc=41
PICTURE_SIZE=32       # CIFAR-10的图片是32x32的
NUM_CLASSES=10        # CIFAR-10的图片是分10类的
PATCH_SIZE=8          # 将一张图片打成8x8的小patch
MLP_HIDDEN_SIZE=512   # Transformer内部的MLP的隐藏层维度
NUM_LAYER=3           # Transformer的层数
STEPLR_STEP_SIZE=20   # 学习率多少个epoch下降一次

EMBEDDING_DIM=24  # q,k,v的长度, 设16则acc=60, 设48则会炸
NUM_HEADS=6                                       # 多头注意力的数量
HIDDEN_SIZE=$((EMBEDDING_DIM * NUM_HEADS))        # Transformer块之间传递的数据维度, = EMDIM * NUM_HEADS
# 6 * 16 ===> 60%
# 6 * 48 ===> x_x
# 6 * 24 ===> 42%
# 8 * 16 ===> 53%
# 8 * 24 ===> 32%
# 6 * 12 ===> 60%
# 4 * 16 ?
# 4 * 12 ?
# 4 * 24 ?
###################################################
# 
###################################################

# 6 * 12
EMBEDDING_DIM=12
NUM_HEADS=6
HIDDEN_SIZE=$((EMBEDDING_DIM * NUM_HEADS))
run

# 4 * 16
EMBEDDING_DIM=16
NUM_HEADS=4
HIDDEN_SIZE=$((EMBEDDING_DIM * NUM_HEADS))
run

# 4 * 12
EMBEDDING_DIM=12
NUM_HEADS=4
HIDDEN_SIZE=$((EMBEDDING_DIM * NUM_HEADS))
run

# 4 * 24
EMBEDDING_DIM=24
NUM_HEADS=4
HIDDEN_SIZE=$((EMBEDDING_DIM * NUM_HEADS))
run


