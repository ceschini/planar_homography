# https://github.com/richzhang/PerceptualSimilarity
# https://www.tutorialspoint.com/how-to-convert-an-image-to-a-pytorch-tensor#

import lpips
import torch
import cv2
import torchvision.transforms as transforms
from torch.nn.functional import normalize
import warnings

# prevent warnings to terminal output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# best forward scores
loss_fn_alex = lpips.LPIPS(net='alex')

# closer to "traditional" perceptual loss, when used for optimization
# loss_fn_vgg = lpips.LPIPS(net='vgg')


# read image
img0 = cv2.imread('../../img/eval_original_image.png')
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

img1 = cv2.imread('../../img/eval_compressed_image1.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


# define transform to convert the image to tensor
transform = transforms.ToTensor()

# convert the image to PyTorch tensor
img0 = transform(img0)
img1 = transform(img1)

# normalize values between -1 and 1
mean0, std0, var0 = torch.mean(img0), torch.std(img0), torch.var(img0)
mean1, std1, var1 = torch.mean(img1), torch.std(img1), torch.var(img1)

img0 = (img0-mean0)/std0
img1 = (img1-mean1)/std1

# passing images to LPIPS model, getting it's output
d = loss_fn_alex.forward(img0, img1)
print(f'Distance between images: {d}')
