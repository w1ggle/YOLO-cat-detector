from ultralytics import YOLO
from heic2png import HEIC2PNG

model = YOLO("best.pt")

#png = HEIC2PNG("IMG_2160.HEIC")
#png.save()

#print(png)

results = model(source = 0, show = False, max_det = 1) #source = 0 means webcam