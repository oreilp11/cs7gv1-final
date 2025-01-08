from ultralytics import YOLO

if __name__ == "__main__":
    baseline_path = "runs/train/pretrained-nano/weights/best.pt"
    model = YOLO(baseline_path)
    model.val(data='dataset.yaml', project='val', name='pretrained-nano', plots=True)

    semodel_path = "runs/train/seyolo-nano/weights/best.pt"
    model = YOLO(semodel_path)
    model.val(data='dataset.yaml', project='val', name='seyolo-nano', plots=True)
