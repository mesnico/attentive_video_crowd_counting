# CAN parameters

PATCH_SIZE_PF = 8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODE = 'concat'
if MODE == 'concat':
    BE_CHANNELS = 1024
elif MODE == 'weighted':
    BE_CHANNELS = 512
