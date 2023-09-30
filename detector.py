from ultralytics import YOLO

model = YOLO("best.pt")

results = model(source = "file1.png", show = True, save = True )