from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="data/raw/dlrs_only_edited.v1i.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    cls=1.5,
    patience=10,
    project="runs/detect",
    name="cinema_occupancy_v1"
)