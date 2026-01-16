
from skimage import io
import torch
from torch.autograd import Variable
from .net_canny import Net
import numpy as np

def canny(raw_img, use_cuda=False):
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()
    print(batch.shape)
    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

    io.imsave('gradient_magnitude.png',grad_mag.data.cpu().numpy()[0,0])
    io.imsave('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
    io.imsave('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
    io.imsave('thresholded.png', early_threshold.data.cpu().numpy()[0, 0])

def canny_batch(data, use_cuda=False):    # batch should be a tensor of shape (batch_size, 3, height, width)
    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()
    if use_cuda:
        data = data.cuda()
    with torch.no_grad():
        blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)
    return grad_mag, thresholded, early_threshold

if __name__ == '__main__':
    # img = io.imread('fb_profile.jpg') / 255.0
    img = np.zeros((256, 256, 3))
    # canny(img, use_cuda=False)
    canny(img, use_cuda=False)
