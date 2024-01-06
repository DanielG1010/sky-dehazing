from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # I add this because the default backend for matplotlib doesn't work in my computer, change it or remove it if you need

config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
# Change cuda:0 to cpu if required
model = init_model(config_file, checkpoint_file, device='cuda:0')

img = 'images/sc_depth_pl_beta0.14054_sample2_frame00104.png'
result = inference_model(model, img)
show_result_pyplot(model, img, result, show=True)


# get the predicted segmentation labels
pred_sem_seg = result.pred_sem_seg.data.cpu().numpy()
pred_sem_seg = np.squeeze(pred_sem_seg)
# create a binary mask for label 2
mask = np.where(pred_sem_seg == 10, 1, 0)

# load the original image
original_image = cv2.imread(img)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)


# invert values of the mask so we have zero where there is sky
inverted_mask = np.where(mask == 0, 1, 0)
inverted_mask_image = np.zeros_like(original_image)
inverted_mask_image[inverted_mask == 1] = [255, 255, 255]

# plot the original image and the mask
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(inverted_mask_image)
plt.title('Mask Image')

plt.show()

