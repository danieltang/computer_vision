import cv2
import numpy as np
from scipy.ndimage.interpolation import geometric_transform
import matplotlib.pyplot as plt
import argparse
import math
import sys
import glob   
import itertools

def binary_it(input_img, thresh):
	output_binary = np.zeros_like(input_img)
	output_binary[(input_img >= thresh[0]) & (input_img <= thresh[1])] = 1
	return output_binary

def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=[0, 255]):

	if orient == 'x':
		sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	elif orient == 'y':
		sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	else:
		print('orient must be either x or y')
		pass

	abs_sobel = np.absolute(sobel)
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	return binary_it(scaled_sobel, thresh)

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

	gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	g_mag = np.sqrt(gx**2+gy**2)

	scaled_g_mag = np.uint8(255*g_mag/np.max(g_mag))
	# print(np.min(g_mag), np.max(g_mag), np.min(scaled_g_mag), np.max(scaled_g_mag))
	return binary_it(scaled_g_mag, thresh)

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):

	sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	abs_sx = np.abs(sx)
	abs_sy = np.abs(sy)

	dir_grad = np.arctan2(abs_sy, abs_sx)

	return binary_it(dir_grad, thresh)

def show_images(cols, rows, images, titles=None, cmap=None):
	f, axarr = plt.subplots(rows, cols, figsize=(20,18))
	if cmap == None:
		cmap = 'gray'

	if rows == 1:
		for iy in range(cols):
			axarr[iy].imshow(images[iy], cmap=cmap)
			axarr[iy].axis('off')
			if titles is not None:
				axarr[iy].set_title(titles[iy], fontsize=12)
			else:
				axarr[iy].set_title("IMG #{}".format(iy), fontsize=12)
	else:
		for ix in range(rows):
			for iy in range(cols):
				i = ix*cols+iy
				axarr[ix, iy].imshow(images[i], cmap=cmap)
				axarr[ix, iy].axis('off')
				if titles is not None:
					axarr[ix, iy].set_title(titles[i], fontsize=12)
				else:
					axarr[ix, iy].set_title("IMG #{}".format(i), fontsize=12)
	plt.show()

def is_circle_homogeneous(img, x0, y0, r0):
	H, W, _ = img.shape
	# x and y coordinates per every pixel of the image
	x, y = np.meshgrid(np.arange(W), np.arange(H))
	# squared distance from the center of the circle
	d2 = (x - x0)**2 + (y - y0)**2
	# mask is True inside of the circle
	mask = d2 < (0.8*r0)**2
	cimg = img[mask, 0]
	if np.std(cimg) < 10.0:
		return True
	else:
		return False

def detect_circle(img, title_str):
	gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
	gray_binary = mag_thresh(gray, sobel_kernel=3, thresh=(1, 80))
	circles = cv2.HoughCircles(gray_binary*255,cv2.HOUGH_GRADIENT,1,20,
	                            param1=50,param2=30,minRadius=20,maxRadius=45)
	xc = -1
	yc = -1
	rc = -1
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			x0 = i[0]
			y0 = i[1]
			r0 = i[2]
			cropped_img = img.copy()
			if is_circle_homogeneous(cropped_img, x0, y0, r0):
				# draw the outer circle
				cv2.circle(img,(x0, y0),r0,(255,0,0),5)
				xc = x0
				yc = y0
				rc = r0
	return (img, gray_binary, [xc, yc, rc])

def detect_lines(background_img, img, minLineLength=6, maxLineGap=8):

	# Apply edge detection method on the image
	add_line_to_img = background_img.copy()
	lines = None
	edges = cv2.Canny(img,50,150,apertureSize = 3)

	lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, lines=np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)

	if lines is not None:
		# The below for loop runs till r and theta values 
		# are in the range of the 2d array
		for line in lines:
			for x1,y1,x2,y2 in line:
				# merge segments
				cv2.line(add_line_to_img,(x1,y1),(x2,y2),(255,0,0),2)

	return add_line_to_img, edges, lines

def filter_lines(center, lines, background_img):
	add_line_to_img = background_img.copy()
	xc, yc, rc = center
	rc2 = rc*rc
	lower_r2 = 2.56*rc2
	upper_r2 = 9.0*rc2
	if lines is not None:
		for line in lines:
			for x1,y1,x2,y2 in line:
				r12 = (x1-xc)**2+(y1-yc)**2
				r22 = (x2-xc)**2+(y2-yc)**2
				# first filter out all outliers
				if (r12-rc2<lower_r2 and r22-rc2<lower_r2) or (r12-rc2>upper_r2 and r22-rc2>upper_r2):
					continue
				# then calculate the distance from center to the line, do determine if it is tangent
				dist = abs((y2-y1)*xc-(x2-x1)*yc+x2*y1-y2*x1)/math.sqrt((y2-y1)**2+(x2-x1)**2)
				if abs(dist-rc)>0.8*rc:
					continue
				cv2.line(add_line_to_img,(x1,y1),(x2,y2),(255,0,0),2)
	return add_line_to_img

def to_polar(img, order=5):
	max_radius = 0.5*np.linalg.norm( img.shape )
	def transform(coords):
		theta = 2.0*np.pi*coords[1] / (img.shape[1] - 1.)
		radius = max_radius * coords[0] / img.shape[0]
		i = 0.5*img.shape[0] - radius*np.sin(theta)
		j = radius*np.cos(theta) + 0.5*img.shape[1]
		return i,j
	polar = geometric_transform(img, transform, order=order,mode='nearest',prefilter=True)
	return polar

def find_histogram_plateau(img):
	w = img.shape[1]
	hist_height = img.shape[0]
	histx = np.nonzero(np.sum(img, axis=1))
	max_histx = int(1.08*np.max(histx)) # add some margin
	h1 = int(0.5*max_histx)
	h2 = int(max_histx)
	histogram = np.sum(img[h1:h2,:], axis=0)
	scaling_factor = (img.shape[0]-1)/np.max(histogram)
	histogram = histogram*scaling_factor

	cv2.line(img,(0,max_histx), (w,max_histx), (255), 2)
	for x,y in enumerate(histogram):
		cv2.rectangle(img,(int(x),int(y)),(int(x+1),hist_height),(127),-1)

	median_line = np.zeros_like(img[0,:])
	median_line[img[max_histx,:]==255]=1
	[l, r] = find_longest_ones(median_line)
	theta = -0.5*(l+r)*2.0*np.pi/w # in radius
	return img, theta

def find_longest_ones(arr):
	z = [(x[0], len(list(x[1]))) for x in itertools.groupby(arr)]
	max_ones = -1
	ind = 0
	islands = [0, 0]
	for (x, xlen) in z:
		ind += xlen
		if x==1 and xlen>max_ones:
			islands[0] = ind
			islands[1] = xlen
			max_ones = xlen
	left_index = islands[0]-islands[1]
	right_index = islands[0]-1
	return (left_index, right_index)

def apply_circle_mask(img, xc, yc, rcutoff):
	rows, cols = img.shape
	# b_img = binary_it(img, [10, 255])*255
	thresh = 0.0175 # 1deg
	for i in range(cols):
		for j in range(rows):
			dy = j-yc
			dx = i-xc
			if math.hypot(dx, dy) > rcutoff:
				img[j,i] = 0
	return img

def crop_image(centerxy, margin, ori_img):
	h1 = (int)(centerxy[0] - margin)
	h2 = (int)(centerxy[0] + margin)
	w1 = (int)(centerxy[1] - margin)
	w2 = (int)(centerxy[1] + margin)
	try:
		img = ori_img[h1:h2, w1:w2, :].copy()
	except:
		img = ori_img[h1:h2, w1:w2].copy()
	return img

def parse_arguments():
	parser = argparse.ArgumentParser(description='Pipeline for heading detection')
	parser.add_argument('-s','--src', help='Source folder to process', required=True)
	args = parser.parse_args()
	return args

def main():
	args = parse_arguments()
	src_folder = args.src

	print('Reading screenshots from source folder {} ...'.format(src_folder))
	path = src_folder + '/*.png'
	files=glob.glob(path)

	# define threshold for blue region in HSV space
	lower_blue = np.array([40,30,150])
	upper_blue = np.array([180,220,255])

	# read first n images and show the blue color thresholds
	index = 0
	ori_demos = []
	direction_demos = []
	road_demos = []
	n_demos = 6
	margin = 300

	for image_file in files:
		image = cv2.imread(image_file)
		(img_h, img_w, _) = image.shape
		# first make a copy of cropped original image to speed up, by assuming blue dot is within center region
		img = crop_image([0.5*img_h, 0.5*img_w], margin, image)
		ori_demos.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		# add blur filter
		img = cv2.GaussianBlur(img, (3,3), 0)
		# Convert BGR to HSV
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
		# Bitwise-AND mask and original image
		hsv_blue = cv2.bitwise_and(hsv, hsv, mask= blue_mask)

		(hsv_blue_circles, hsv_blue_circles_binary, center) = detect_circle(hsv_blue.copy(), image_file)

		original_center = center.copy()

		if (center[0]<=0 or center[1]<=0):
			print("No circle detected for {} ...".format(image_file))
			continue

		half_margin = margin//2
		# crop image based on detected circle center
		h_channel = crop_image([center[1], center[0]], half_margin, hsv_blue[:,:,0])
		cimg = crop_image([center[1], center[0]], half_margin, img)
		# update center coords
		center[0] = center[1] = half_margin

		# show_images(2,1,[hsv_blue, h_channel])

		# further process with h channel
		h_channel = cv2.medianBlur(h_channel, 5)
		h_channel = apply_circle_mask(h_channel, center[0], center[1], 3.0*center[2])

		cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)

		## method 1: detecting lines
		# detect lines and filter non-tangiential ones
		_, _, lines = detect_lines(cimg, h_channel)
		direction_marked_img = filter_lines(center, lines, cimg)

		## method 2: histogram
		# tranform image from cartersian to polar, and filter plateau on histogram
		polar_img = to_polar(h_channel)
		polar_img, theta = find_histogram_plateau(polar_img)
		cv2.line(direction_marked_img, (center[0], center[1]), (center[0]+int(5.0*center[2]*np.cos(theta)), center[1]+int(5.0*center[2]*np.sin(theta))), (255,0,0),4)

		direction_demos.append(direction_marked_img)

		# start with cropped img
		# crop img to center the circle
		img = crop_image([original_center[1], original_center[0]], half_margin, img)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		b_channel = cv2.bitwise_not(binary_it(cv2.GaussianBlur(img[:,:,2],(3,3),5), [210, 250]))*255
		b_mag_binary = mag_thresh(b_channel, sobel_kernel=3, thresh=(200, 300))*255
		img_with_lines, edges, lines = detect_lines(img, b_mag_binary, 60, 10)
		road_demos.append(img_with_lines)

		index += 1
		if index >= n_demos:
			break
	n_cols = 3
	show_images(n_cols, n_demos//n_cols, ori_demos)
	show_images(n_cols, n_demos//n_cols, direction_demos)
	show_images(n_cols, n_demos//n_cols, road_demos)

if __name__ == '__main__':
	main()