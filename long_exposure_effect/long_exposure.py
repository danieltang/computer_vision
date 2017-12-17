# import the necessary packages
import argparse
import cv2

def apply_long_exposure_effect(input_video, output_img):

	# initialize the Red, Green, and Blue channel averages, along with# the total number of frames read from the file
	channels = None
	frames = 0

	print("Opening video file ...")
	stream = cv2.VideoCapture(input_video)
	print("Computing frame averages (this will take awhile)...")

	# loop over frames from the video file stream
	while True:
		# grab the frame from the file stream
		(grabbed, frame) = stream.read()
		# if the frame was not grabbed, then we have reached the end of the file	
		if not grabbed:
			break
		# otherwise, split the frmae into its respective channels
		current_channels = cv2.split(frame.astype("float"))
		n_channels = len(current_channels)
		# if the frame averages are None, initialize them	
		if channels is None:
			channels = current_channels
		# otherwise, compute the weighted average between the history of frames and the current frames	
		else:
			for k in range(n_channels):
				channels[k] = ((frames * channels[k]) + (1 * current_channels[k])) / (frames + 1.0)
		# increment the frame number by one
		frames += 1

	# merge the RGB averages together and write the output image to disk
	avg = cv2.merge(channels).astype("uint8")
	cv2.imwrite(output_img, avg)
	# do a bit of cleanup on the file pointer
	stream.release()

def parse_arguments():
	# construct the argument parse and parse the argumentsap = argparse.ArgumentParser()
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--video", required=True, help="path to input video file")
	parser.add_argument("-o", "--output", required=True, help="path to output png file with long exposure effect")
	args = parser.parse_args()

	return args

def main():
	args = parse_arguments()
	apply_long_exposure_effect(args.video, args.output)

if __name__ == '__main__':
    main()
