import sys
sys.path.append('core')
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import os

velocity_list=[]
masked_motion_vectors = None
DEVICE = 'cuda'
motion_vectors=None
# Create an empty list to store images
count=0
image_list = []
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)
def compute_velocity(motion_vectors, scaling_factor):
    magnitude = motion_vectors[:, :, 0]
    velocity = magnitude * scaling_factor
    return velocity
def visualize_motion_vectors(motion_vectors):
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
    # Plot the motion vectors as arrows
    # plt.figure(figsize=(10, 10))
    # plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
    # plt.axis('image')
    # Rotate the plot by 180 degrees
    # plt.gca().invert_yaxis()
    # plt.show()
    velocity_list = np.append(velocity_list, vel_final)
    print("value",np.mean(velocity_list))
def viz(img, flo):
    global count
    global image_list
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    motion_vectors=flow_viz.flow_to_motion_vectors(flo)
    flo = flow_viz.flow_to_image(flo)
    
    img_flo = np.concatenate([img, flo], axis=1)

    # cv2.imshow('image',img_flo[:, :, [2, 1, 0]] / 255.0)
    x,y,w,h=mask_image(img)
    motion_vectors1=motion_vectors[y:y + h,x:x + w, :]
    visualize_motion_vectors(motion_vectors1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

    # Draw bounding box on the image for the maximum area
    if max_box is not None:
        x, y, w, h = max_box
        label = "Car"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Car Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return x,y,w,h
    # cv2.waitKey()
def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
      
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
