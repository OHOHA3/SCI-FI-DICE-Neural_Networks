from src.functions import func_module as fm
import cv2

cap = cv2.VideoCapture(0)
model = fm.load_model()

while True:
    fm.make_image(cap)
    inputs = fm.image_processing()
    predicted_class = fm.predict_class(model, inputs)
    fm.send_result(predicted_class)
