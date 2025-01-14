from ultralytics import YOLO

if __name__ == "__main__":
    baseline_path = "runs/train/pretrained-nano/weights/best.pt"
    model = YOLO(baseline_path)
    model.val(data='dataset.yaml', split='test', project='test', name='pretrained-nano', plots=True)

    semodel_path = "runs/train/seyolo-nano/weights/best.pt"
    model = YOLO(semodel_path)
    model.val(data='dataset.yaml', split='test', project='test', name='seyolo-nano', plots=True)