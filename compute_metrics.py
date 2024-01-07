import os
import csv
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# Define the directories
dir1 = 'gt_new'
dir2 = 'gt_new'

# Get the list of image names in the directories
image_names_dir1 = os.listdir(dir1)
image_names_dir2 = os.listdir(dir2)

# Initialize lists to store the PSNR and SSIM values
psnr_values = []
ssim_values = []

# Loop over each image in the first directory
for image_name_dir1 in tqdm(image_names_dir1, desc="Processing images"):
    # Find the matching image in the second directory
    for image_name_dir2 in image_names_dir2:
        # Check if the starting string of the image names match
        if image_name_dir1.split('_')[0] == image_name_dir2.split('_')[0]:
            # Load the images
            img1 = img_as_float(io.imread(os.path.join(dir1, image_name_dir1)))
            img2 = img_as_float(io.imread(os.path.join(dir2, image_name_dir2)))
            
            # Normalize the images to the range [0, 1]
            img1 = img_as_float(img1)
            img2 = img_as_float(img2)
 
            # Calculate the PSNR and SSIM
            psnr_value = psnr(img1, img2)
            ssim_value = ssim(img1, img2, channel_axis=-1, data_range=1.0)

            # Append the values to the lists
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
        
            # Break the loop as we have found the matching image
            break

# Calculate the average PSNR and SSIM
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

# Print the average values
print(f'Average PSNR: {avg_psnr}')
print(f'Average SSIM: {avg_ssim}')

# Write the values to a CSV file
with open('image_metrics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Pair", "PSNR", "SSIM"])
    writer.writerows(zip(image_names_dir1, psnr_values, ssim_values))
    writer.writerow(["Average", avg_psnr, avg_ssim])

