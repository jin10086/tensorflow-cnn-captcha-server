import string
import os

# 验证码目录
DATA_DIR = r"/Users/gaojin/Downloads/sougou_com_Trains"

# 图片的shape
# 可以通过 cv2.imread().shape 拿到,理论上验证码图片的大小都是一样的,如果不一样,需要resize成一样的
# 验证码的高,宽,层
H, W, C = 66, 203, 3


str_charts = string.digits + string.ascii_letters  # 验证码里的所有字符
N_LABELS = len(str_charts)
D = 6  # 验证码长度 (比如4位数的验证码,6位数的等等 )

# 训练时保存的文件夹

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# 到多少准确度以后就停止训练
accuracy_rate = 0.95
model_file_name = "mymodel.h5"
