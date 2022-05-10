# import the necessary packages
from imutils.video import FPS
import multiprocessing
import numpy as np
import argparse
import imutils
import dlib     # <----- for correlation-based tracking algorithm
import cv2

import obj_detector as ODT
import config

def get_params():
    global model_path
    global input_img_path
    global DETECTION_THRESHOLD
    global frame_path

    params=config.get_params()

    model_path = params["model_path"]
    input_img_path = params["img_path"]
    DETECTION_THRESHOLD = params["detect_threshold"]
    frame_path = params["frame_path"]


def start_tracker(box, label, rgb, inputQueue, outputQueue):
	# construct a dlib rectangle object from the bounding box
	# coordinates and then start the correlation tracker
	t = dlib.correlation_tracker()
	rect = dlib.rectangle(box[0], box[1], box[2], box[3])
	t.start_track(rgb, rect)

	# loop indefinitely -- this function will be called as a daemon
	# process so we don't need to worry about joining it
	while True:
		# attempt to grab the next frame from the input queue
		rgb = inputQueue.get()

		# if there was an entry in our queue, process it
		if rgb is not None:
			# update the tracker and grab the position of the tracked
			# object
			t.update(rgb)
			pos = t.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the label + bounding box coordinates to the output
			# queue
			outputQueue.put((label, (startX, startY, endX, endY)))


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", #required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# initialize our lists of queues -- both input queue and output queue
# for *every* object that we will be tracking


# initialize the list of class labels on which model was trained to
# detect
CLASSES = ["1head", "1tail", "5head", "5tail","10head", "10tail"]


def main():
    inputQueues = []
    outputQueues = []
    # initialize the video stream and output video writer
    print("[INFO] starting video stream...")
    #vs = cv2.VideoCapture(args["video"])
    vs = cv2.VideoCapture(0)
    writer = None

    # start the frames per second throughput estimator
    fps = FPS().start()

    # loop over frames from the video file stream
    while True:
        # grab the next frame from the video file
        (grabbed, frame) = vs.read()

        # check to see if we have reached the end of the video file
        if frame is None:
            break

        # resize the frame for faster processing and then convert the
        # frame from BGR to RGB ordering (dlib needs RGB ordering)
        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imwrite(frame_path, frame)



        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)

        # if our list of queues is empty then we know we have yet to
        # create our first object tracker
        if len(inputQueues) == 0:
            # grab the frame dimensions and convert the frame to a blob
            (h, w) = frame.shape[:2]

            # call object detection and get the detected bbox.
            # results is list of dictionaries
            results = ODT.main(model_path, frame_path, DETECTION_THRESHOLD)


            for result in results:
                # filter out weak detections by requiring a minimum
                if result["score"] > DETECTION_THRESHOLD:
                    # Extract the coordinates of the bounding boxes
                    ymin, xmin, ymax, xmax = result["bounding_box"]
                    ymin = int(ymin * h)
                    xmin = int(xmin * w)
                    ymax = int(ymax * h)
                    xmax = int(xmax * w)
                    bb = (xmin, ymin, xmax, ymax)

                    # extract the class label from the detection list
                    class_index = result["class_id"]
                    print(int(class_index))
                    label = CLASSES[int(class_index)]
                    print(label)

                    # create two brand new input and output queues,
                    # respectively
                    iq = multiprocessing.Queue()
                    oq = multiprocessing.Queue()
                    inputQueues.append(iq)
                    outputQueues.append(oq)

                    # spawn a daemon process for a new object tracker
                    p = multiprocessing.Process(
                        target=start_tracker,
                        args=(bb, label, rgb, iq, oq))
                    p.daemon = True
                    p.start()

                    # grab the corresponding class label for the detection
                    # and draw the bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                        (0, 255, 0), 2)
                    #cv2.putText(frame, label, (xmin, ymin - 15),
                     #   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # otherwise, we've already performed detection so let's track
        # multiple objects
        else:
            # loop over each of our input ques and add the input RGB
            # frame to it, enabling us to update each of the respective
            # object trackers running in separate processes
            for iq in inputQueues:
                iq.put(rgb)

            # loop over each of the output queues
            for oq in outputQueues:
                # grab the updated bounding box coordinates for the
                # object -- the .get method is a blocking operation so
                # this will pause our execution until the respective
                # process finishes the tracking update
                (label, (xmin, ymin, xmax, ymax)) = oq.get()

                # draw the bounding box from the correlation object
                # tracker
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                    (0, 255, 0), 2)
                #cv2.putText(frame, label, (xmin, ymin - 15),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()

if __name__=="__main__":
    get_params()
    main()