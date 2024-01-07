import numpy as np
import matplotlib.pyplot as plt
import dezero
import dezero.functions as F
import dezero.layers as L
from dezero import DataLoader
from dezero.models import Sequential
from dezero.optimizers import Adam
from matplotlib.animation import PillowWriter
from _rich import new_progress
import math


metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = PillowWriter(fps=15, metadata=metadata)
fig = plt.figure()
progress = new_progress()

use_gpu = dezero.cuda.gpu_enable
max_epoch = 5
batch_size = 128
hidden_size = 62

fc_channel, fc_height, fc_width = 128, 7, 7

gen = Sequential(
    L.Linear(1024),
    L.BatchNorm(),
    F.relu,
    L.Linear(fc_channel * fc_height * fc_width),
    L.BatchNorm(),
    F.relu,
    lambda x: F.reshape(x, (-1, fc_channel, fc_height, fc_width)),
    L.Deconv2d(fc_channel // 2, kernel_size=4, stride=2, pad=1),
    L.BatchNorm(),
    F.relu,
    L.Deconv2d(1, kernel_size=4, stride=2, pad=1),
    F.sigmoid
)

dis = Sequential(
    L.Conv2d(64, kernel_size=4, stride=2, pad=1),
    F.leaky_relu,
    L.Conv2d(128, kernel_size=4, stride=2, pad=1),
    L.BatchNorm(),
    F.leaky_relu,
    F.flatten,
    L.Linear(1024),
    L.BatchNorm(),
    F.leaky_relu,
    L.Linear(1),
    F.sigmoid
)


def init_weight(dis, gen, hidden_size):
    # Input dummy data to initialize weights
    batch_size = 1
    z = np.random.rand(batch_size, hidden_size)
    fake_images = gen(z)
    dis(fake_images)

    for l in dis.layers + gen.layers:
        classname = l.__class__.__name__
        if classname.lower() in ('conv2d', 'linear', 'deconv2d'):
            l.W.data = 0.02 * np.random.randn(*l.W.data.shape)

init_weight(dis, gen, hidden_size)

opt_g = Adam(alpha=0.0002, beta1=0.5).setup(gen)
opt_d = Adam(alpha=0.0002, beta1=0.5).setup(dis)

transform = lambda x: (x / 255.0).astype(np.float32)
train_set = dezero.datasets.MNIST(train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size)

if use_gpu:
    gen.to_gpu()
    dis.to_gpu()
    train_loader.to_gpu()
    xp = dezero.cuda.cupy
else:
    xp = np

label_real = xp.ones(batch_size).astype(np.int_)
label_fake = xp.zeros(batch_size).astype(np.int_)
test_z = xp.random.randn(25, hidden_size).astype(np.float32)


def generate_image(test_z=None, images=None, writer=None, savefig=False, showIsNumber=False):
    with dezero.test_mode():
        if images is None:
            images = gen(test_z)
        y_fake = dis(images)

    img = dezero.cuda.as_numpy(images.data)
    len_img = len(images)
    row = math.ceil(math.sqrt(len_img))
    col = math.ceil(len_img / row)
    fig.clear()
    fig.suptitle('Is Number: 0.00%')

    for i in range(0, len_img):
        ax = fig.add_subplot(row, col, i+1)
        ax.axis('off')
        if showIsNumber:
            ax.set_title('%.2f%%' % (y_fake[i][0].data * 100), fontsize=16)
        ax.imshow(img[i][0], 'gray')

    if writer is not None:
        writer.grab_frame()

    # if savefig:
    #     plt.savefig('gan_{}.png'.format(idx))


def main():
    progress.start()
    task_id = progress.add_task('epoch: 0', total=max_epoch)

    for epoch in range(max_epoch):
        avg_loss_d = 0
        avg_loss_g = 0
        cnt = 0
        max_iter = train_loader.max_iter

        subtask_id = progress.add_task(f'iter {cnt}/{max_iter}', total=max_iter)
    
        for x, t in train_loader:
            cnt += 1
            if len(t) != batch_size:
                continue

            # (1) Update discriminator
            z = xp.random.randn(batch_size, hidden_size).astype(np.float32)
            fake = gen(z)
            y_real = dis(x)
            y_fake = dis(fake.data)
            loss_d = F.binary_cross_entropy(y_real, label_real) + \
                     F.binary_cross_entropy(y_fake, label_fake)
            gen.cleargrads()
            dis.cleargrads()
            loss_d.backward()
            opt_d.update()
    
            # (2) Update generator
            y_fake = dis(fake)
            loss_g = F.binary_cross_entropy(y_fake, label_real)
            gen.cleargrads()
            dis.cleargrads()
            loss_g.backward()
            opt_g.update()
    
            # Print loss & visualize generator
            avg_loss_g += loss_g.data
            avg_loss_d += loss_d.data
            interval = 100 if use_gpu else 5
            if cnt % interval == 0:
                epoch_detail = epoch + cnt / max_iter
                print('epoch: {:.2f}, loss_g: {:.4f}, loss_d: {:.4f}'.format(
                    epoch_detail, float(avg_loss_g/cnt), float(avg_loss_d/cnt)))
                generate_image(writer=writer)

            progress.update(subtask_id, advance=1, description=f'iter {cnt}/{max_iter}')

        progress.remove_task(subtask_id)
        progress.update(task_id, advance=1, description=f'epoch: {epoch+1}')

    progress.remove_task(task_id)


if __name__ == '__main__':
    # with writer.saving(fig, "gan.gif", 100):
    #     main()

    # gen.save_weights('gan.npz')
    # dis.save_weights('dis.npz')

    gen.load_weights('gan.npz')
    dis.load_weights('dis.npz')

    n_sample = 25
    x = xp.random.randn(n_sample, hidden_size).astype(np.float32)

    # generate_image(x, showIsNumber=True)
    train_loader = DataLoader(train_set, batch_size=n_sample, shuffle=True)

    for x, t in train_loader:
        generate_image(images=x, showIsNumber=True)
        plt.show()
        break
