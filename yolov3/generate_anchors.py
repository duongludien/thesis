import glob
import os
import utils
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


TRAIN_DIR = '/home/diendl/Desktop/new_dataset/data/train'
TEST_DIR = '/home/diendl/Desktop/new_dataset/data/test'
NUM_ANCHORS = 9
WIDTH = 416
HEIGHT = 416

label_files = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
label_files += glob.glob(os.path.join(TEST_DIR, '*.txt'))

X = None
for label_path in label_files:
    true_boxes = utils.read_label(label_path)
    for true_box in true_boxes:
        bw, bh = true_box[3] * WIDTH, true_box[4] * HEIGHT
        this_box = np.expand_dims([bw, bh], axis=0)
        if X is None:
            X = this_box
        else:
            X = np.concatenate((X, this_box), axis=0)

kmeans = KMeans(n_clusters=NUM_ANCHORS, random_state=0).fit(X)
centers = np.array(kmeans.cluster_centers_)
print(centers)

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'brown', 'olive', 'pink']

for item in X:
    result = kmeans.predict(np.expand_dims(item, 0))[0]
    plt.scatter(x=item[0], y=item[1], c=colors[result])

plt.scatter(x=centers[:, 0], y=centers[:, 1], c='black', marker='x')
plt.xlabel('width (bw)')
plt.ylabel('height (bh)')
plt.show()
