# from roboflow import Roboflow
#
# rf = Roboflow(api_key="KAgjDOo9HinEQDDxfazo")
# project = rf.workspace("kt-nyjuw").project("floor-plan-nnoub")
# version = project.version(1)
# dataset = version.download("yolov8")
#

from ultralytics import YOLO

model = YOLO('yolov8l.pt')

model.train(data='C:/Users/ACER/PycharmProjects/YOLO-Object-Detection-Course/Projects/floor-plan-1/data.yaml', epochs=50, imgsz=640)
