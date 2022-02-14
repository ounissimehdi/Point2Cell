#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:29:36 2021

@author: mehdi.ounissi
@email : mehdi.ounissi@icm-institue.org
		 mehdi.gtx@gmail.com
"""
from skimage.measure import regionprops
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd
import distancemap as dm
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2, os
import skimage.measure
from glob import glob
from natsort import natsorted
from shutil import copyfile


def get_file_extention(path):
    split_path = os.path.normpath(path).lstrip(os.path.sep).split(os.path.sep)
    return os.path.splitext(split_path[-1])[1]

def annotate_img(img_path, mask_path, dist_path):
	global center_list
	def click_and_crop(event, x, y, flags, param):
		# grab references to the global variables
		global center_list
	
		if event == cv2.EVENT_LBUTTONDOWN:
			# record the (x, y) coordinates
			center_list.append((x, y))
	
			# draw a cyrcle in the center
			cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
			cv2.imshow("image", image)
	
	# initialize the list of (x, y) coordinates
	center_list = []
	
	# load the image, clone it, and setup the mouse callback function
	image = cv2.imread(img_path)
	mask = np.array(Image.open(mask_path))
	dist = np.array(Image.open(dist_path))
	
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	
	# Initi the flag
	flag = True
	
	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(0) & 0xFF
		
		# if the 'r' key is pressed, reset all clicks
		if key == ord("r"):
			image = clone.copy()
			center_list = []
		
		# if the 's' key is pressed, delete the last click
		elif key == ord("s"):
			image = clone.copy()
			center_list.pop()
			for (x, y) in center_list:
				cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
				cv2.imshow("image", image)
		
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			cv2.destroyWindow("image")
			break

		# if the 'q' key is pressed, break from the loop then close all
		elif key ==ord("q"):
			flag = False
			break
	
	# close all open windows
	cv2.destroyAllWindows()
	
	if flag:
		x_list, y_list = [], []
		mask_out = np.zeros(dist.shape, dtype=bool)
		for i in range(len(center_list)):
			x = int(center_list[i][0])
			y = int(center_list[i][1])
			
			mask_out[y,x] = True

			x_list.append(x)
			y_list.append(y)

		markers, _ = ndi.label(mask_out)
		labels = watershed(-dist, markers, mask=mask>0.2)

		#plt.imshow(labels, cmap = plt.cm.nipy_spectral)
		#plt.show()
		
		new_mask = np.zeros(mask.shape, dtype="uint8")
		new_dist = np.zeros(mask.shape, dtype="uint8")
		
		for label in np.unique(labels):
			if label == 0: continue
			
			temp_mask = np.zeros(mask.shape, dtype="uint8")
			temp_mask[labels == label] = 255
			
			temp_mask = ndi.binary_erosion(temp_mask, structure=np.ones((2,2))).astype(temp_mask.dtype)
			temp_mask = ndi.binary_fill_holes(temp_mask)
			
			rr, cc = np.where(temp_mask==True)
			new_mask[rr, cc] = 255
		
		
		labels = skimage.measure.label(new_mask)
		
		new_dist = np.zeros(mask.shape, dtype="uint8")
		for region in regionprops(labels):
			
			cord = np.array(region.centroid,dtype=np.int64)
			
			dm_mask = np.zeros(mask.shape).astype(bool)
			dm_mask[cord[0],cord[1]] = True
		
		
			gauss_dist = dm.distance_map_from_binary_matrix(dm_mask,distance="euclidean", alpha="linear")
		
			new_dist[labels == region.label] = 255 - gauss_dist[labels == region.label]
		
		labels_out = Image.fromarray(new_mask)
	
		
		dist_out = Image.fromarray(new_dist)
	
		return labels_out, dist_out, x_list, y_list
	return 0, 0, [], []
# The in files
in_files = os.path.join('..', '..', 'dataset', 'hela_cells_dataset', 'novel_annotation')

# Where to put the annotated data
out_files = os.path.join('..', '..', 'dataset', 'hela_cells_dataset', 'novel_annotation')

# Creating folder to store the selected data
os.makedirs(os.path.join(out_files,'refined_masks'), exist_ok = True)
os.makedirs(os.path.join(out_files,'refined_density_maps'), exist_ok = True)

# Getting all the imgs inside the in files
all_imgs_files = natsorted(glob(os.path.join(in_files, "images",'*.tif')))

# Getting all the pre-annotated data imgs inside the in files
annotated_imgs = natsorted(glob(os.path.join(out_files, "refined_masks",'*.tif')))

# The list of indx of the pre-annotated data
idx_old = []
for path in annotated_imgs:
	split_path = os.path.normpath(path).lstrip(os.path.sep).split(os.path.sep)
	img_name = os.path.splitext(split_path[-1])[0]
	idx_old.append(img_name)

try:
	old_data_frame = pd.read_csv(os.path.join(out_files, "cells_centroids_GT.csv"))
	x_list_csv, y_list_csv, frame_csv = list(old_data_frame['x']), list(old_data_frame['y']), list(old_data_frame['frame_name'])
	print("[INFO] GT cells centroids csv file found appending old data to the new annotations ...")
except FileNotFoundError:
	x_list_csv, y_list_csv, frame_csv = [], [], []
	print("[INFO] No pervious csv file for GT cells centroids making a new one...")
	
for img_path in tqdm(all_imgs_files):
	split_path = os.path.normpath(img_path).lstrip(os.path.sep).split(os.path.sep)
	img_name = os.path.splitext(split_path[-1])[0]
	img_ext = os.path.splitext(split_path[-1])[1]
	if any(img_name in element for element in idx_old): continue
	else:

		# Check if the dist map is available
		dist_check = glob(os.path.join(in_files, "density_masks",img_name+'.*'))

		# Check if the cell is available
		mask_check = glob(os.path.join(in_files, "binary_masks",img_name+'.*'))

		if len(mask_check) != 0 and len(dist_check) != 0:
			dist_ext = get_file_extention(dist_check[0])
			mask_ext = get_file_extention(mask_check[0])
			
			# init validation flag
			validation = "n"
			while validation =="n":
				labels_out, dist_out, x_list_tmp, y_list_tmp = annotate_img(img_path, mask_check[0], dist_check[0])
				
				if labels_out !=0 and dist_out !=0:
					w, h = labels_out.size
					
					resized_img = Image.open(img_path)
					
					resized_img = resized_img.resize((w, h))
					
					plt.figure(figsize=(15,5))
					plt.subplot(131)
					plt.imshow(labels_out)
					plt.scatter(x_list_tmp, y_list_tmp, s = 60, c = 'red', marker = '*', edgecolors = 'white')
					plt.axis('off')
					plt.tight_layout()
					
					plt.subplot(132)
					plt.imshow(resized_img)
					plt.scatter(x_list_tmp, y_list_tmp, s = 60, c = 'red', marker = '*', edgecolors = 'white')
					plt.axis('off')
					plt.tight_layout()
					
					plt.subplot(133)
					plt.imshow(dist_out)
					plt.scatter(x_list_tmp, y_list_tmp, s = 60, c = 'red', marker = '*', edgecolors = 'white')
					plt.axis('off')
					plt.tight_layout()
					plt.show()
					
					validation = input("[INFO] Is the segmentation okay [yes/no/skip] ? (y/n/s)") or "y"
					
					if validation == "y":
						for ii in range(len(x_list_tmp)):
							x_list_csv.append(x_list_tmp[ii])
							y_list_csv.append(y_list_tmp[ii])
							frame_csv.append(int(img_name))
					
						csv_details = {'frame_name': frame_csv,'x': x_list_csv,'y': y_list_csv}

						# Writing data in a csv file
						data_frame = pd.DataFrame(csv_details)
						
						# Save the csv file 
						data_frame.to_csv(os.path.join(out_files, "cells_centroids_GT.csv"), index = False)


						# copyfile(img_path, os.path.join(out_files,'original_imgs',img_name+img_ext) )
						labels_out.save(os.path.join(out_files,'refined_masks',img_name+mask_ext))
						dist_out.save(os.path.join(out_files,'refined_density_maps',img_name+dist_ext))
						# resized_img.save(os.path.join(out_files,'imgs',img_name+img_ext))
	end = input("[INFO] You want it to end here ? (y/n)") or "n"
	if end=="y": break
	


