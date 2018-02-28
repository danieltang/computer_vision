import cv2
import numpy as np
from scipy.ndimage.interpolation import geometric_transform
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import argparse
import math
import sys
import glob   
import itertools
from operator import itemgetter

arc2deg = 180.0/math.pi

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
	return binary_it(scaled_g_mag, thresh)

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):

	sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	abs_sx = np.abs(sx)
	abs_sy = np.abs(sy)

	dir_grad = np.arctan2(abs_sy, abs_sx)

	return binary_it(dir_grad, thresh)

def show_images(cols, rows, images, titles=None, cmap=None):
	n_imgs = len(images)
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
				if i >= n_imgs:
					continue
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
	xc, yc, rc = -1, -1, -1
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
	if img is not None:
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

def cluster_lines(lines, center):
	# assume lines are already sorted by distance from center, and segment length
	(r, c) = lines.shape
	start_idx = 0
	end_idx = 0
	clusters = []
	for k in range(r):
		if end_idx+1>r-1:
			break
		if (abs(lines[end_idx+1,0]-lines[start_idx,0])<0.1 and abs(lines[end_idx+1,2]-lines[start_idx,2])<10):
			end_idx += 1
			if end_idx >= r-1:
				clusters.append([start_idx, end_idx])
		else:
			if end_idx>=start_idx:
				clusters.append([start_idx, end_idx])
			else:
				clusters.append([start_idx, start_idx])

			start_idx = k+1
			end_idx = start_idx

	clustered_lines = []
	for c in clusters:
		xx = np.concatenate([lines[c[0]:(c[1]+1), 3], lines[c[0]:(c[1]+1), 5]])
		yy = np.concatenate([lines[c[0]:(c[1]+1), 4], lines[c[0]:(c[1]+1), 6]])
		left, top, right, bottom = np.min(xx), np.min(yy), np.max(xx), np.max(yy)
		# extend x1, y1, x2, y2
		# pick up the first representive segment
		x1, y1, x2, y2 = lines[c[0], 3], lines[c[0], 4], lines[c[0], 5], lines[c[0], 6]
		dx = x2-x1
		dy = y2-y1
		scaling_factor = 100
		dx = dx*scaling_factor
		dy = dy*scaling_factor
		# extend by scaling factor
		xx1 = x2 + dx
		yy1 = y2 + dy
		xx2 = x2 - dx
		yy2 = y2 - dy
		clipped_line = liangbarsky(left, top, right, bottom, xx1, yy1, xx2, yy2)
		if (clipped_line[0] is None):
			continue
		clustered_lines.append(clipped_line)
	return clustered_lines

def liangbarsky(xmin, ymin, xmax, ymax, x1, y1, x2, y2):
	# defining variables
	p1 = -(x2 - x1)
	p2 = -p1
	p3 = -(y2 - y1)
	p4 = -p3

	q1 = x1 - xmin
	q2 = xmax - x1
	q3 = y1 - ymin
	q4 = ymax - y1

	posarr=[]
	negarr=[]
	posarr.append(1)
	negarr.append(0)

	if ((p1 == 0 and q1 < 0) or (p3 == 0 and q3 < 0)):
		print("Line is parallel to clipping window!")
		return None

	if (p1 != 0):
		r1 = q1 / p1
		r2 = q2 / p2
		if (p1 < 0):
			negarr.append(r1) # for negative p1, add it to negative array
			posarr.append(r2) # and add p2 to positive array
		else:
			negarr.append(r2)
			posarr.append(r1)

	if (p3 != 0):
		r3 = q3 / p3
		r4 = q4 / p4
		if (p3 < 0):
			negarr.append(r3)
			posarr.append(r4)
		else:
			negarr.append(r4)
			posarr.append(r3)

	rn1 = np.max(negarr) # maximum of negative array
	rn2 = np.min(posarr) # minimum of positive array

	xn1 = int(x1 + p2 * rn1)
	yn1 = int(y1 + p4 * rn1) # computing new points
	xn2 = int(x1 + p2 * rn2)
	yn2 = int(y1 + p4 * rn2)

	return xn1, yn1, xn2, yn2

def find_primary_road(lines, center):
	# first find the longest lines
	angle_thresh = 2.0*np.pi/180
	dist_thresh = 10
	ordered_lines = []
	if lines is not None:
		for k in range(len(lines)):
			for x1,y1,x2,y2 in lines[k]:
				dx = x2 - x1
				dy = y2 - y1
				seg_length = math.sqrt(dx*dx+dy*dy)
				dist_from_origin = abs(x2*y1-y2*x1)/seg_length
				dist_from_center = abs(dy*center[0]-dx*center[1]+x2*y1-y2*x1)/seg_length
				theta = np.arctan2(dy, dx)
				ordered_lines.append([theta, dist_from_origin, dist_from_center, x1, y1, x2, y2, seg_length, k])
	sorted_lines = sorted(ordered_lines, key=lambda x: x[7])[-16:] # first sort with seg length
	sorted_lines = sorted(ordered_lines, key=lambda x: x[2])[:16] # then sort with dist_from_origin
	clustered_lines = cluster_lines(np.array(sorted_lines), center)

	# print("current lines")
	# print(sorted_lines)
	primary_line = []
	for s in clustered_lines:
		dx = s[2]-s[0]
		dy = s[3]-s[1]
		segl2 = dx*dx+dy*dy
		if segl2 < 400:
			continue
		primary_line = s
		break
	return primary_line

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
	direction_demos = []
	road_demos = []
	n_demos = len(files)
	margin = 300
	results = []
	for image_file in files:
		image = cv2.imread(image_file)
		(img_h, img_w, _) = image.shape
		# first make a copy of cropped original image to speed up, by assuming blue dot is within center region
		img = crop_image([0.5*img_h, 0.5*img_w], margin, image)
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

		half_margin = int(margin/1.5)
		# crop image based on detected circle center
		h_channel = crop_image([center[1], center[0]], half_margin, hsv_blue[:,:,0])
		cimg = crop_image([center[1], center[0]], half_margin, img)
		# update center coords
		center[0] = center[1] = half_margin

		# further process with h channel
		h_channel = cv2.medianBlur(h_channel, 5)
		h_channel = apply_circle_mask(h_channel, center[0], center[1], 3.0*center[2])

		cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)

		## method 1: detecting lines
		# detect lines and filter non-tangiential ones
		if cimg is None or h_channel is None:
			print("Reading Error")
			continue

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
		img2 = img.copy()
		hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		l_channel = binary_it(cv2.GaussianBlur(hls_img[:,:,1],(3,3),5), [240, 255])
		# filter out island with a threshold of num_of_pixels_thresh
		labels = label(l_channel)
		num_of_pixels_thresh = 1000
		for idx in range(1, labels[1] + 1):
			# Find pixels with each label value
			nonzero = (labels[0] == idx).nonzero()
			nonzerox = np.array(nonzero[0])
			n_size = len(nonzerox)
			if n_size < num_of_pixels_thresh:
				labels[0][nonzero] = 0
			else:
				labels[0][nonzero] = 1
		l_channel = labels[0].copy()
		l_channel = np.float32(l_channel)
		l_mag_binary = mag_thresh(l_channel, sobel_kernel=3, thresh=(200,300))*255

		b_channel = cv2.bitwise_not(binary_it(cv2.GaussianBlur(img[:,:,2],(3,3),5), [210, 250]))*255
		b_mag_binary = mag_thresh(b_channel, sobel_kernel=3, thresh=(200, 300))*255

		img_with_lines0, edges, lines = detect_lines(img, l_mag_binary, 40, 6)
		primary_line = find_primary_road(lines, center)
		road_theta = np.arctan2(primary_line[3]-primary_line[1], primary_line[2]-primary_line[0])
		e1 = abs(np.arctan2(np.sin(road_theta-theta), np.cos(road_theta-theta)))
		e2 = abs(np.arctan2(np.sin(np.pi+road_theta-theta), np.cos(np.pi+road_theta-theta)))
		wrapped_error = np.min([e1, e2])
		cv2.line(direction_marked_img, (primary_line[0], primary_line[1]), (primary_line[2], primary_line[3]), (0,0,255),4)

		results.append([index, image_file, int(theta*arc2deg), int(road_theta*arc2deg), int(wrapped_error*arc2deg)])
		index += 1

		if index >= n_demos:
			break
	titles = []
	for r in results:
		print(r)
		titles.append("IMG # {}, Heading Error {}".format(r[0], r[4]))

	n_cols = 6
	show_images(n_cols, (index-1)//n_cols+1, direction_demos, titles)

if __name__ == '__main__':
	main()