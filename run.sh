cd src
function run() {
    python main.py \
        --epoch=$EPOCH \
        --batch_size=$BATCH_SIZE \
        --hidden_size=$HIDDEN_SIZE \
        --patch_size=$PATCH_SIZE \
        --feedforward_dim=$FEEDFORWARD_DIM \
        --num_layers=$NUM_LAYERS \
        --num_heads=$NUM_HEADS \
}


EPOCH=250             # epoch数量
BATCH_SIZE=256        # batch大小  # 64 acc=41
PATCH_SIZE=8          # 将一张图片打成8x8的小patch
FEEDFORWARD_DIM=512   # Transformer内部的MLP的隐藏层维度
NUM_LAYERS=3          # Transformer的层数

HIDDEN_SIZE=96        # q,k,v的长度, 设16则acc=60, 设48则会炸
NUM_HEADS=6           # 多头注意力的数量

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




