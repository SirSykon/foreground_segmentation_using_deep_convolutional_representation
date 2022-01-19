import evaluation_utils
import numpy as np

a = np.zeros(shape=(3,3,3))
a[0] = 255.
a[2,1] = 255.
print(a)

gt = np.ones(shape=(3,3,3))*255.
gt[2] = 0
print(gt)

print(evaluation_utils.compare_gt_and_img(a, gt))
