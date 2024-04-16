from transformers import DetrImageProcessor, DetrForObjectDetection
from src.functions import func_module as fm
import sys
import cv2

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
cap = cv2.VideoCapture(0)
room = int(sys.argv[1])
time_zone = int(sys.argv[2])

while True:
    fm.make_image(cap)
    boxes = fm.find_people(processor, model)
    count = fm.make_bounds(boxes)
    fm.send_result(room, count, time_zone)
