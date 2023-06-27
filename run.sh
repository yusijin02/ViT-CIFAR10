cd src
EPOCH=600             # epoch数量
BATCH_SIZE=256        # batch大小  # 64 acc=41
PICTURE_SIZE=32       # CIFAR-10的图片是32x32的
NUM_CLASSES=10        # CIFAR-10的图片是分10类的
PATCH_SIZE=8          # 将一张图片打成8x8的小patch
MLP_HIDDEN_SIZE=512   # Transformer内部的MLP的隐藏层维度
NUM_LAYER=3           # Transformer的层数
STEPLR_STEP_SIZE=20   # 学习率多少个epoch下降一次

EMBEDDING_DIM=16                             # q,k,v的长度
NUM_HEADS=6                                  # 多头注意力的数量
HIDDEN_SIZE=$EMBEDDING_DIM*$NUM_HEADS        # Transformer块之间传递的数据维度, = EMDIM * NUM_HEADS
###################################################
# run ViT in CIFAR-10
###################################################
python main.py  \
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




EPOCH=600             # epoch数量
BATCH_SIZE=256        # batch大小  # 64 acc=41
PICTURE_SIZE=32       # CIFAR-10的图片是32x32的
NUM_CLASSES=10        # CIFAR-10的图片是分10类的
PATCH_SIZE=8          # 将一张图片打成8x8的小patch
MLP_HIDDEN_SIZE=512   # Transformer内部的MLP的隐藏层维度
NUM_LAYER=3           # Transformer的层数
STEPLR_STEP_SIZE=20   # 学习率多少个epoch下降一次

EMBEDDING_DIM=48                             # q,k,v的长度
NUM_HEADS=6                                  # 多头注意力的数量
HIDDEN_SIZE=$EMBEDDING_DIM*$NUM_HEADS        # Transformer块之间传递的数据维度, = EMDIM * NUM_HEADS
###################################################
# run ViT in CIFAR-10
###################################################
python main.py  \
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
