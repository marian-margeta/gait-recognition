import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread

from human_pose_nn import HumanPoseIRNetwork
mpl.use('Agg')

net_pose = HumanPoseIRNetwork()
net_pose.restore('../Thesis_solution/models/MPII+LSP.ckpt')

img = imread('images/dummy.jpg')
img = imresize(img, [299, 299])
img_batch = np.expand_dims(img, 0)

y, x, a = net_pose.estimate_joints(img_batch)
y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)

joint_names = [
    'right ankle ',
    'right knee ',
    'right hip',
    'left hip',
    'left knee',
    'left ankle',
    'pelvis',
    'thorax',
    'upper neck',
    'head top',
    'right wrist',
    'right elbow',
    'right shoulder',
    'left shoulder',
    'left elbow',
    'left wrist'
]

# Print probabilities of each estimation
for i in range(16):
    print('%s: %.02f%%' % (joint_names[i], a[i] * 100))

colors = ['r', 'r', 'b', 'm', 'm', 'y', 'g', 'g', 'b', 'c', 'r', 'r', 'b', 'm', 'm', 'c']
for i in range(16):
    if i < 15 and i not in {5, 9}:
        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color = colors[i], linewidth = 5)

plt.imshow(img)
plt.savefig('images/dummy_pose.jpg')