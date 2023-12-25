import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from blazeface import BlazeFace

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnt = [0]

def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=detections[i, 16])
                ax.add_patch(circle)
        
    cnt[0] += 1
    plt.savefig("detection_" + str(cnt[0]) + ".png")

################################
# Front detections
################################
front_net = BlazeFace()
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")

# Optionally change the thresholds:
front_net.min_score_thresh = 0.75
front_net.min_suppression_threshold = 0.3

img = cv2.imread("1face.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

front_detections = front_net.predict_on_image(img)
plot_detections(img, front_detections)

#################################
# Back detections
#################################
back_net = BlazeFace(back_model=True)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

img2 = cv2.resize(img, (256, 256))

back_detections = back_net.predict_on_image(img2)
plot_detections(img2, back_detections)

#################################
# Predict on batch of images
#################################
filenames = [ "1face.png", "3faces.png", "4faces.png" ]

xback = np.zeros((len(filenames), 256, 256, 3), dtype=np.uint8)

for i, filename in enumerate(filenames):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    xback[i] = cv2.resize(img, (256, 256))

back_detections = back_net.predict_on_batch(xback)
plot_detections(xback[0], back_detections[0])
plot_detections(xback[1], back_detections[1])
plot_detections(xback[2], back_detections[2])
