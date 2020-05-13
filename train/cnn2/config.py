import string
import os

# 验证码目录
DATA_DIR = r"/Users/gaojin/Downloads/12306captcha/12306_pic"

# 图片的shape
# 可以通过 cv2.imread().shape 拿到,理论上验证码图片的大小都是一样的,如果不一样,需要resize成一样的
# 验证码的高,宽,层
H, W, C = 68, 67, 3


# 训练时保存的文件夹

checkpoint_path = "training_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# 到多少准确度以后就停止训练
accuracy_rate = 0.95
model_file_name = "12306.h5"

# 图片后缀
image_type = "jpg"
