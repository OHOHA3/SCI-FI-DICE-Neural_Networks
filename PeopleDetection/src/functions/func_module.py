from datetime import datetime, timezone, timedelta
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import torchvision.transforms as tvt
from PIL import Image
import requests
import torch
import cv2

CAMERA_IMAGE = "src/camera_pictures/camera_image.png"
BOUNDED_IMAGE = "src/camera_pictures/bounded_image.png"
SERVER_URL = "http://37.230.195.102:8083/neuronet/setPeople"


def make_image(cap):
    for i in range(30):
        cap.read()
    ret, frame = cap.read()
    cv2.imwrite(CAMERA_IMAGE, frame)


def find_people(processor, model):
    with Image.open(CAMERA_IMAGE) as camera_image:
        inputs = processor(images=camera_image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([camera_image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        boxes = [box.tolist() for label, box in zip(results["labels"], results["boxes"]) if
                 model.config.id2label[label.item()] == "person"]
    return boxes


def make_bounds(boxes):
    count = len(boxes)
    if count > 0:
        tensor_image = read_image(CAMERA_IMAGE)
        boxes = torch.tensor(boxes, dtype=torch.float)
        tensor_result = draw_bounding_boxes(tensor_image, boxes, width=6, colors="green")
        bounded_image = tvt.ToPILImage()(tensor_result)
        bounded_image.save(BOUNDED_IMAGE)
    else:
        with Image.open(CAMERA_IMAGE) as img:
            img.save(BOUNDED_IMAGE)
    return count


def send_result(room, number, time_zone):
    utc_time = datetime.now(timezone.utc)
    current_time = utc_time + timedelta(hours=time_zone)
    current_time = current_time.strftime("%D, %H:%M:%S")
    print(f"{number} people in {room} room, date: {current_time}")
    url = f"{SERVER_URL}?room={room}&people={number}&time={current_time}"
    with open(BOUNDED_IMAGE, "rb") as picture:
        files = {"image": picture}
        try:
            requests.post(url, files=files)
        except requests.exceptions.ConnectionError:
            print("No connection to server")
