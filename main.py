import cv2
import numpy as np
import time
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
import matplotlib.pyplot as plt
import csv
from utils import intersects, on_mouse, scale
import os


# Path to the video file. Leave blank to use the webcam.
VIDEO_PATH = 'traffic.mp4'
# Where to write the resulting video. Leave blank to not write a video.
WRITE_PATH = 'results.avi'
# Define a font, scale and thickness to be used when displaying class names.
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
# Define a confidence threshold for box detections and a threshold for non-max suppression.
CONF_THRESH = 0.5
NMS_THRESH = 0.4
# Define the max cosine distance and budget of the nearest neighbor distance metric used to associate the
# appearance feature vectors of new detections and existing tracks.
MAX_COSINE_DISTANCE = 0.5
NN_BUDGET = None
# Define the number of frames for class name smoothing (the class name detected most
# often in the last N frames will be displayed above the bounding box for each object)
N_NAMES_SMOOTHING = 30
# Define the coordinates of the lines representing reach road's entry/exit.
ROAD_LINES = [[(620, 155), (499, 349)],
              [(296, 55), (613, 141)],
              [(47, 137), (239, 64)],
              [(470, 379), (27, 149)]]
# Define the names of the respective roads from COUNT_LINES.
ROAD_NAMES = ['road 1',
              'road 2',
              'road 3',
              'road 4']

# Create and open a CSV file where information about the detections will be logged.
csvfile = open('results.csv', 'w', newline='')
# Define the names of the columns which will be used to make entries in each column.
fieldnames = ['Id', 'Type', 'From', 'Towards']
# Create a writer object for the CSV file with the fieldnames and write them on the first row.
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

# Delete all .png files, which might have been added by previously running the script,
# from the "counted_vehicles" folder.
filelist = [f for f in os.listdir("counted_vehicles") if f.endswith(".png")]
for f in filelist:
    os.remove(os.path.join("counted_vehicles", f))

# Load the YOLOv3 model with OpenCV.
net = cv2.dnn.readNet("yolov3/yolov3.weights", "yolov3/yolov3.cfg")
# Get the names of all layers in the network.
layer_names = net.getLayerNames()
# Extract the names of the output layers by finding their indices in layer_names.
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialise a list to store the classes names.
classes = []
# Set each line in "coco.names" to an entry in the list, stripping whitespace.
with open("yolov3/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialise the encoder used to create appearance feature vectors of detections.
encoder = gdet.create_box_encoder("deep_sort/mars-small128.pb", batch_size=1)
# Initialise the nearest neighbor distance metric.
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
# Initialise a Tracker object with the defined metric and frames of class name smoothing.
tracker = Tracker(metric, n_names_smoothing=N_NAMES_SMOOTHING)

# If the video path is not empty, initialise a video capture object with that path.
if VIDEO_PATH != '':
    cap = cv2.VideoCapture(VIDEO_PATH)
# Otherwise, initialise a video capture object with the first camera.
else:
    cap = cv2.VideoCapture(0)

# If the write path is not empty, initialise a video writer object for that path with the same dimensions and FPS as
# the video capture object, using the XVID video codec.
if WRITE_PATH != '':
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(WRITE_PATH, codec, fps, (width, height))

# Initialise a color map.
cmap = plt.get_cmap("gist_rainbow")
# Initialise a colors matrix with dimensions (classes, colors per class, color channels)
colors = np.zeros((80, 3, 3))
# Define the step size used to split the colormap into 80 regions.
n = 1/80
# Iterate over the number of classes.
for i in range(len(classes)):
    # Grab 3 colors from the i-th region in the color map and store them as the i-th entry in the color matrix.
    colors[i] = [cmap(j)[:3] for j in np.linspace(0+n*i, n+n*i, 3)]
# Scale the values in the colors matrix to be between 0 and 255.
colors *= 255
# Shuffle the first dimension of the color matrix, to avoid having very similar colors for similar classes that often
# appear together. This is done because the classes in the coco dataset are sorted by similarity.
np.random.shuffle(colors)

# Initialise a frame counter and get the current time for FPS calculation purposes.
frame_id = 0
time_start = time.time()

# Initialise a sting used for displaying a log message for the last recorded entry/exit of a vehicle.
log_text = ""

while True:
    # Read the current frame from the camera.
    _, frame = cap.read()
    # Check if the video has ended. If it has, break the loop.
    if frame is None:
        break
    # Add 1 to the frame count every time a frame is read.
    frame_id += 1
    # Get the shape of the unprocessed frame.
    height, width, channels = frame.shape

    # Pre-process the frame by applying the same scaling used when training the YOLO model, resizing to the size
    # expected by this particular YOLOv3 model, and swapping from BGR (used by OpenCV) to RGB (used by the model).
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True)

    # Pass the processed frame through the neural network to get a prediction.
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialise arrays for storing confidence, coordinate values and class names for detected boxes.
    confidences = []
    boxes = []
    names = []

    # Loop through all the detections in each of the three output scales of YOLOv3.
    for out in outs:
        for detection in out:
            # Get the class probabilities for this box detection.
            scores = detection[5:]
            # Find the class with the highest score for the box.
            class_id = np.argmax(scores)
            # Extract the score of that class.
            confidence = scores[class_id]
            # If that score is higher than the set threshold, execute the code in the "if" statement.
            if confidence > CONF_THRESH:
                # Use the predicted box ratios to get box coordinates which apply to the input image.
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Use the center, width and height coordinates to calculate the coordinates for the top left
                # point of the box, which is the required format for Deep SORT.
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                # Populate the arrays with the information for this box.
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                names.append(str(classes[class_id]))

    # Apply non-max suppression to get rid of overlapping boxes.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    NMSBoxes = [boxes[int(i)] for i in indexes]

    # Compute appearance features of the detected boxes.
    features = encoder(frame, NMSBoxes)
    # Create a list of detection objects.
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(np.array(NMSBoxes), np.array(confidences), np.array(names), np.array(features))]

    # Predict the current state of existing tracks using a Kalman filter.
    tracker.predict()
    # Update the tracker by using the current detections and predictions from the Kalman filter.
    tracker.update(detections)

    # Iterate over existing tracks in the tracker.
    for track in tracker.tracks:
        # If the current track is not confirmed or the number of frames since the last measurement update of the track
        # are more than one, skip this track. (Only confirmed tracks for the current frame pass through)
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        # If the class is not one of the ones we are interested in, skip this track.
        if not track.get_class() in ('car', 'motorbike', 'bus', 'truck'):
            continue
        # Get the bounding box from the detection object in the format (min x, min y, max x, max y)
        tl_x, tl_y, br_x, br_y = track.to_tlbr().astype(int)
        # Get the class name for the current track.
        class_name = track.get_class()
        # Find the class ID of the track.
        class_id = classes.index(class_name)
        # Select one of the three colors associated with that class based on the track ID.
        color = colors[class_id, int(track.track_id) % 3]
        # Draw the bounding box around the object.
        cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), color, 2)

        # Define the text to be displayed above the main box.
        text = class_name + " " + str(track.track_id)
        # Get the width and height of the text to place a filled box behind it.
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        # Draw the filled box where the text would be.
        cv2.rectangle(frame, (tl_x, tl_y), (tl_x + text_width, tl_y - text_height - 15), color, -1)
        # Draw the text on top of the box.
        cv2.putText(frame, text, (tl_x, tl_y - 10), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

        # Get the line segment defined by the centre of the track's bounding box from
        # the previous frame and from the current one, and scale it by a factor of 5.
        scaled_line = scale(track.line, 2)

        # Iterate through the defined road lines.
        for i in range(len(ROAD_LINES)):
            # If a road line intersects with the scaled track line, execute the "if" statement.
            if intersects(ROAD_LINES[i], scaled_line):
                # If this track has already been recorded as passing through this line, move to the next iteration.
                if ROAD_NAMES[i] in (track.approach, track.exit):
                    continue
                # Tell the track object that the track has passed through this line.
                track.passed_line(ROAD_NAMES[i])
                # If the exit road of the track is empty, then the road line it just intersected corresponds
                # to the road that the track entered the intersection from, so set log_text to this string.
                if track.exit == '':
                    log_text = "{} {} entered from {}".format(class_name, track.track_id, track.approach)
                # Otherwise, the road line it just intersected corresponds to the road that the track
                # exited the intersection from, so execute the code in the "else" statement below.
                else:
                    # Set log_text to this string.
                    log_text = "{} {} exited from {}".format(class_name, track.track_id, track.exit)
                    # Write a new row to the CSV file with the information for this track.
                    writer.writerow({'Id': track.track_id, 'Type': class_name, 'From': track.approach, 'Towards': track.exit})
                    # Get a clean version of the current frame, with no extra drawing on top.
                    _, clean_frame = cap.read()
                    # Get the image inside the bounding box of the track.
                    cropped_img = clean_frame[tl_y:br_y, tl_x:br_x]
                    # Write that image in the counted_vehicles folder with
                    # a name containing the vehicle type and track Id.
                    cv2.imwrite(r'counted_vehicles\{}_{}.png'.format(class_name, track.track_id), cropped_img)

        # Uncomment the line below if you wish for the scaled lines used for detection to be displayed.
        cv2.line(frame, scaled_line[0], scaled_line[1], (255, 255, 255), 2)

    # Draw a filled box where the log_text will be displayed.
    cv2.rectangle(frame, (width - 260, height - 20), (width, height), (1, 1, 1), -1)
    # Draw the log_text on top of the box.
    cv2.putText(frame, log_text, (width - 255, height - 5), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

    # Display the defined road lines and road names.
    for i in range(len(ROAD_LINES)):
        cv2.line(frame, ROAD_LINES[i][0], ROAD_LINES[i][1], (200, 200, 0), 2)
        # To avoid clutter, an Id is displayed for each road instead of its name.
        # Replace "str(i + 1)" with "ROAD_NAMES[i]" to display the names of the roads instead.
        cv2.putText(frame, str(i + 1), ROAD_LINES[i][0], FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

    # If the write path is not blank, write the frame to the output video.
    if WRITE_PATH != '':
        output.write(frame)

    # Calculate the elapsed time since starting the loop.
    elapsed_time = time.time() - time_start
    # Calculate the average FPS performance to this point.
    fps = frame_id/elapsed_time
    # Display the FPS at the top left corner.
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (8, 30), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    # Show the frame.
    cv2.imshow("Video", frame)
    # Display mouse coordinates in the console when clicking on the video. Useful for determining line coordinates.
    cv2.setMouseCallback('Video', on_mouse)
    # Wait at least 1ms for key event and record the pressed key.
    key = cv2.waitKey(1)
    # If the pressed key is ESC (27), break the loop.
    if key == 27:
        break

# Release the capture object and destroy all windows.
cap.release()
cv2.destroyAllWindows()
# Close the CSV file.
csvfile.close()
