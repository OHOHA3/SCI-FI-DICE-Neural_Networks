import cv2
import transforms as t
import os

transform = t.transform6()

in_directory = "full"
out_directory = "dataset\\test\\full"
index = ""
i = 1
for filename in os.listdir(in_directory):
    image = cv2.imread(in_directory + "\\" + filename)
    #t.resolution(i, in_directory + "\\" + filename)
    #transformed_image = cv2.resize(image, (256, 256))
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    cv2.imwrite(out_directory + "\\" + str(i) + index + ".png", transformed_image)
    i += 1
