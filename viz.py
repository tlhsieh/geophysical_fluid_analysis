import numpy as np
import imageio

def crop(filename, outfilename='cropped.png'):
    """Remove white borders around an image"""

    image = imageio.imread(filename)

    # find indices of non-white pixels
    shape = image.shape
    gray_scale = np.mean(image, axis=2) # collapsing the RGB axis

    ax0 = np.mean(gray_scale, axis=1) # collapsing axis 1, leaving only axis 0
    idx_ax0_beg = np.argmax(ax0 < 255)
    idx_ax0_end = shape[0] - np.argmax(np.flip(ax0) < 255)

    ax1 = np.mean(gray_scale, axis=0) # collapsing axis 0, leaving only axis 1
    idx_ax1_beg = np.argmax(ax1 < 255)
    idx_ax1_end = shape[1] - np.argmax(np.flip(ax1) < 255)

    cropped = image[idx_ax0_beg:idx_ax0_end, idx_ax1_beg:idx_ax1_end, :]
    imageio.imsave(outfilename, cropped)

def concat_img(filenames, outfilename='combined', axis=0):
    """Combine a list of images to one. Input images can have different sizes. Output has the same extension as input.

    Example:
        concat_img(['input1.png', 'input2.png'], 'output', 1)

    Args:
        axis: 0 for vertical stack; 1 for horizontal strip
    """

    images = [imageio.imread(filename) for filename in filenames]
    maxax0 = np.max([image.shape[0] for image in images])
    maxax1 = np.max([image.shape[1] for image in images])

    padded = []
    for image in images:
        white = np.ones([maxax0, maxax1, image.shape[2]], dtype=image.dtype)*255
        white[:image.shape[0], :image.shape[1], :] = image
        padded.append(white)

    combined = np.concatenate(padded, axis=axis)
    imageio.imsave(outfilename+'.'+filenames[0].split('.')[-1], combined)
