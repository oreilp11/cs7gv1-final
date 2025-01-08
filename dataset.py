import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

LABELS = [
    'sheep',
    'cattle',
    'seal',
    'camelus',
    'kiang',
    'zebra',
]
num_labels = len(LABELS)
base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'WAID', 'labels')

class_distributions = {
    'all': Counter({label:0 for label in LABELS}),
    'train': Counter({label:0 for label in LABELS}),
    'valid': Counter({label:0 for label in LABELS}),
    'test': Counter({label:0 for label in LABELS})
    }

num_occurrences = []

for dataset in ['train', 'valid', 'test']:
    train_path = os.path.join(base_path, dataset)
    for filename in tqdm(os.listdir(train_path)):
        with open(os.path.join(train_path, filename)) as file:
            classes = [LABELS[int(line.split(" ")[0])] for line in file.readlines()]
        class_distributions[dataset].update(classes)
        class_distributions['all'].update(classes)
        num_occurrences.append(len(classes))

for dataset in class_distributions:
    total = sum(class_distributions[dataset].values())
    class_distributions[dataset] = {label:value/total for label, value in class_distributions[dataset].items()}
print(f"Average no. species per image - mean:{np.mean(num_occurrences):0.4f}, std:{np.std(num_occurrences):0.4f}")
class_distributions = pd.DataFrame(class_distributions)
print(class_distributions)

class_distributions.plot.bar(y=['train','valid','test'])
plt.title("Class Distribution Proportions")
plt.xlabel("Classes")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig("plots/classdist")
plt.show()

baseline_train_loss = "runs/train/pretrained-nano/results.csv"
baseline = pd.read_csv(baseline_train_loss)
plt.figure(figsize=(6,6))
plt.subplot(3,1,1)
plt.title("Baseline YOLOv10 Box Loss", fontdict={})
plt.plot(baseline['epoch'][5:], baseline['train/box_loss'][5:], label='train')
plt.plot(baseline['epoch'][5:], baseline['val/box_loss'][5:], label='val')
plt.ylabel("Box Loss")
plt.legend()

plt.subplot(3,1,2)
plt.title("Baseline YOLOv10 Class Loss")
plt.plot(baseline['epoch'][5:], baseline['train/cls_loss'][5:], label='train')
plt.plot(baseline['epoch'][5:], baseline['val/cls_loss'][5:], label='val')
plt.ylabel("Class Loss")
plt.legend()

plt.subplot(3,1,3)
plt.title("Baseline YOLOv10 DFL Loss")
plt.plot(baseline['epoch'][5:], baseline['train/dfl_loss'][5:], label='train')
plt.plot(baseline['epoch'][5:], baseline['val/dfl_loss'][5:], label='val')
plt.xlabel("Epochs")
plt.ylabel("DFL Loss")
plt.legend()
plt.tight_layout()
plt.savefig("plots/pretrainloss.png")
plt.show()

seyolo_train_loss = "runs/train/seyolo-nano/results.csv"
seyolo = pd.read_csv(seyolo_train_loss)
plt.figure(figsize=(6,6))
plt.subplot(3,1,1)
plt.title("SE-YOLOv10 Box Loss")
plt.plot(seyolo['epoch'][5:], seyolo['train/box_loss'][5:], label='train')
plt.plot(seyolo['epoch'][5:], seyolo['val/box_loss'][5:], label='val')
plt.ylabel("Box Loss")
plt.legend()

plt.subplot(3,1,2)
plt.title("SE-YOLOv10 Class Loss")
plt.plot(seyolo['epoch'][5:], seyolo['train/cls_loss'][5:], label='train')
plt.plot(seyolo['epoch'][5:], seyolo['val/cls_loss'][5:], label='val')
plt.ylabel("Class Loss")
plt.legend()

plt.subplot(3,1,3)
plt.title("SE-YOLOv10 DFL Loss")
plt.plot(seyolo['epoch'][5:], seyolo['train/dfl_loss'][5:], label='train')
plt.plot(seyolo['epoch'][5:], seyolo['val/dfl_loss'][5:], label='val')
plt.xlabel("Epochs")
plt.ylabel("DFL Loss")
plt.legend()
plt.tight_layout()
plt.savefig("plots/seyololoss.png")
plt.show()