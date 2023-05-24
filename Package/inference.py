import os
import sys
sys.path.append('core')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import OrderedDict

import cv2
import numpy as np
import torch

from raft import RAFT
from utils import flow_viz

velocity_list=[]

def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame
def compute_velocity(motion_vectors, scaling_factor):
    magnitude = motion_vectors[:, :, 0]
    velocity = magnitude * scaling_factor
    return velocity
def vizualize_flow(img, flo, save, counter):
    # permute the channels and change device is necessary
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    motion_vectors=flow_viz.flow_to_motion_vectors(flo)
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
    x,y,w,h=mask_image(img)
    motion_vectors1=motion_vectors[y:y + h,x:x + w, :]
    visualize_motion_vectors(motion_vectors1,img,w)
    # concatenate, save and show images
    img_flo = np.concatenate([img, flo], axis=1)
    cv2.imshow("Optical Flow", img_flo / 255.0)
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        return False
    return True
def visualize_motion_vectors(motion_vectors,img,w):
    global velocity_list
    magnitude = motion_vectors[:, :, 0]
    direction = motion_vectors[:, :, 1]
    # Create a grid of coordinates for the motion vectors
    y, x = np.mgrid[0:magnitude.shape[0], 0:magnitude.shape[1]]
    # Calculate the arrow displacements
    dx = magnitude * np.cos(direction)
    dy = magnitude * np.sin(direction)
    sensor_size_mm=3.674
    frame_rate=30
    image_width_in_pixels=640
    conversion_factor = sensor_size_mm / image_width_in_pixels
    # Calculate the scaling factor by incorporating the temporal aspect
    scaling_factor = conversion_factor * frame_rate
    velocities = compute_velocity(motion_vectors, scaling_factor)
    vel_final=np.linalg.norm(velocities)
    print(vel_final)
    # # Plot the motion vectors as arrows
    # plt.figure(figsize=(10, 10))
    # plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
    # plt.axis('image')
    # # Rotate the plot by 180 degrees
    # plt.gca().invert_yaxis()
    # plt.show()
    velocity_list = np.append(velocity_list, vel_final)
    print("value",np.mean(velocity_list))
    
def mask_image(image):
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

    # Specify the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    image = cv2.convertScaleAbs(image, 1.0 / 255.0)

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run inference
    outputs = net.forward(output_layers)

    # Get bounding box information
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 2:  # Assuming class_id 2 represents cars
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])

                # Calculate top-left corner coordinates of the bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # Find the maximum bounding box
    max_area = 0
    max_box = None
    for i in indices:
        i = i.item()
        x, y, w, h = boxes[i]
        area = w * h

        if area > max_area:
            max_area = area
            max_box = boxes[i]
    x=0
    y=0
    w=0
    h=0
    # Draw bounding box on the image for the maximum area
    if max_box is not None:
        x, y, w, h = max_box
    # Draw bounding box rectangle and label
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("Car Detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return x,y,w,h
    # cv2.waitKey()

def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def inference(args):
    # get the RAFT model
    model = RAFT(args)
    # load pretrained weights
    pretrained_weights = torch.load(args.model)

    save = args.save
    if save:
        if not os.path.exists("demo_frames"):
            os.mkdir("demo_frames")

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
        model = torch.nn.DataParallel(model)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)
        model.to(device)
    else:
        device = "cpu"
        # change key names for CPU runtime
        pretrained_weights = get_cpu_model(pretrained_weights)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)

    # change model's mode to evaluation
    model.eval()

    video_path = args.video
    # capture the video and get the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame_1 = cap.read()

    # frame preprocessing
    frame_1 = frame_preprocess(frame_1, device)

    counter = 0
    with torch.no_grad():
        while True:
            # read the next frame
            ret, frame_2 = cap.read()
            if not ret:
                break
            # preprocessing
            frame_2 = frame_preprocess(frame_2, device)
            # predict the flow
            flow_low, flow_up = model(frame_1, frame_2, iters=args.iters, test_mode=True)
            # transpose the flow output and convert it into numpy array
            ret = vizualize_flow(frame_1, flow_up, save, counter)
            if not ret:
                break
            frame_1 = frame_2
            counter += 1


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--video", type=str, default="1684366668.0944755.mp4")
    #for live video streaming from the bot
    # parser.add_argument("--video", type=str, default="ttp://192.168.137.185:8000/stream.mjpg")
    parser.add_argument("--save", action="store_true", help="save demo frames")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
