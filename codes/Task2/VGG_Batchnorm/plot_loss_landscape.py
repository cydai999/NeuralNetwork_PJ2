import numpy as np
import matplotlib.pyplot as plt

lrs = [1e-3, 2e-3, 1e-4, 5e-4]
vgg_loss_list = []
vgg_bn_loss_list = []

for lr in lrs:
    with open(f"log/v3/vgg/loss_{lr}.txt", 'r') as f:
        loss_list = f.read().replace('\n', ' ').strip().split(' ')
        loss_list = [float(loss) for loss in loss_list]
        vgg_loss_list.append(np.array(loss_list))
    with open(f"log/v3/vgg_bn/loss_{lr}.txt") as f:
        loss_list = f.read().replace('\n', ' ').strip().split(' ')
        loss_list = [float(loss) for loss in loss_list]
        vgg_bn_loss_list.append(np.array(loss_list))

vgg_loss = np.stack(vgg_loss_list, 0)
vgg_bn_loss = np.stack(vgg_bn_loss_list, 0)

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
vgg_min_curve = np.min(vgg_loss, 0)
vgg_max_curve = np.max(vgg_loss, 0)
vgg_bn_min_curve = np.min(vgg_bn_loss, 0)
vgg_bn_max_curve = np.max(vgg_bn_loss, 0)

# plot the final loss landscape
def plot_loss_landscape(axis, min_curve, max_curve, color, label):
    axis.plot(min_curve, color=color, label=label)
    axis.plot(max_curve, color=color)
    axis.fill_between(range(len(min_curve)), min_curve, max_curve, color=color)

fg, ax = plt.subplots()
plot_loss_landscape(ax, vgg_min_curve, vgg_max_curve, color='#ADDF9B', label='vgg')
plot_loss_landscape(ax, vgg_bn_min_curve, vgg_bn_max_curve, color='#D16D60', label='vgg_bn')
plt.ylim((0, 2.5))
plt.legend()
plt.show()
# plt.savefig('v3.png')

