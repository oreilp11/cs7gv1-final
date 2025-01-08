import os
import argparse

from ultralytics import YOLO


if __name__ == "__main__":
    #model = YOLO("yolov10n.pt")
    #model.train(data='dataset.yaml', batch=-1, epochs=100, device='cuda', val=True, save=True, verbose=True)

    model = YOLO("seyolo10n.yaml")
    model.train(data='dataset.yaml', batch=-1, epochs=100, device='cuda', val=True, save=True, verbose=True)
