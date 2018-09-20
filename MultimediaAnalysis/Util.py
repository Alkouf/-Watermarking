import numpy as np
import random
import time
from PIL import Image, ImageFilter


# returns numpy array
def create_watermark(key, length):
    # uses MT19937
    # print rng.get_state()
    rng = np.random.RandomState(key)

    r = np.zeros(length, dtype=float)
    for l in range(0, length):
        if rng.uniform(low=0.0, high=1.0) - 0.5 > 0:
            r[l] = 1
        else:
            r[l] = -1

    # subtract the mean from the vector
    mean = r.mean()
    # print "original mean:"+ str(mean)
    r = r - mean
    # print "new mean:"+ str(r.mean())
    return r


# random chunks
def create_chunks(chunks, length):
    random.seed(int(time.time() * 100))
    f = np.zeros(shape=(chunks, length), dtype=float)
    for c in range(0, chunks):
        for l in range(0, length):
            f[c][l] = random.randint(0, 256)
    return f


# detects the "correlation" of signal f with the watermark
# returns a value that should be equal to p (embedding power), if the watermark exists
def detect(f, w):
    # inner product of f,w
    # print "f[0]size : " + str(f.shape[0])
    d = 0.0
    for c in range(0, f.shape[0]):
        d += np.inner(f[c], w)
    d /= f.shape[0]
    d /= (np.linalg.norm(w) ** 2)
    return d


# embeds the given watermark in the signal f accordingly to the embedding power "power"
def embed_watermark_signum(f, w, power):
    for c in range(0, f.shape[0]):
        for j in range(0, len(f[c])):
            if f[c][j] >= 0:
                f[c][j] += w[j] * power  # * np.abs(f[c][j])
            else:
                print "ARNITIKO OEOPOEOOEOEOE"
                f[c][j] -= w[j] * power  # * np.abs(f[c][j])
    return f


# Creates an array of length (= argument length)
# with noise given from a Gaussian distribution, with mean=0, and sigma=magnitude
# (White noise?)
def create_noise(length, magnitude):
    r = np.zeros(length, dtype=float)
    for l in range(0, length):
        r[l] = random.gauss(mu=0, sigma=magnitude)
    return r


def f_from_mask(arr, length, mask):
    # print mask.shape
    # print arr.shape
    if mask.shape != arr.shape:
        print "Wrong array shapes, exit!!"
        return
    nofpixel = np.sum(mask)
    chunks = int(nofpixel) / int(length)
    f = np.zeros(shape=(chunks, length), dtype=float)
    index = 0
    c = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if mask[i][j] == 1:
                f[c][index] = arr[i, j]
                index += 1
                if index == length:
                    index = 0
                    c += 1
                if c == chunks:
                    return f
    return f


def f_to_mask(f, arr, length, mask):
    if mask.shape != arr.shape:
        print "Wrong array shapes, exit!!"
        return
    arr2 = arr.copy()
    chunks = f.shape[0]
    index = 0
    c = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if mask[i][j] == 1:
                arr2[i, j] = f[c][index]
                index += 1
                if index == length:
                    index = 0
                    c += 1
                if c == chunks:
                    return arr2
    return arr2

# Applies Gaussian white noise to all image pixels, with sigma = magnitude.
def image_noise(im, magnitude):
    im2 = im.copy()
    pixels = im2.load()
    for i in range(im2.size[0]):
        for j in range(im2.size[1]):
            p = pixels[i, j] + int(random.gauss(mu=0, sigma=magnitude))
            if p < 0:
                p = 0
            if p > 255:
                p = 255
            pixels[i, j] = p
    return im2


# Resizes the image
def image_resize(im, percentage):
    return im.resize(size=(int(im.size[0] * percentage), int(im.size[1] * percentage)))


# left, upper, right, and lower pixel
def image_cropping(im, padding):
    im2 = im.copy()
    im2.load()
    if type(padding) == type(100):
        if im2.size[1] > 2 * padding:
            im2 = im2.crop((0, padding, im.size[0], im.size[1] - padding))
        if im2.size[0] > 2 * padding:
            im2 = im2.crop((padding, 0, im.size[0] - padding, im.size[1]))
    if type(padding) == type(10.0):
        width, height = im.size  # Get dimensions
        left = width * padding / 2
        top = height * padding / 2
        right = width - left
        bottom = height - top
        # print "l,t,r,b :" + str(left) + " , " + str(top) + " , " + str(right) + " , " + str(bottom)
        im2 = im.crop((left, top, right, bottom))
    return im2


def image_rotate(im, degrees):
    return im.rotate(degrees)


def image_blur(im):
    return im.filter(ImageFilter.BLUR)


def image_median(im, w=3):
    return im.filter(ImageFilter.MedianFilter(size=w))


# returns an PIL image object from the given array
# I think the array can have real values
# append '.show' to show the image
def arr2image(arr):
    if len(arr[arr[:] > 255]) > 0:
        print "THERE ARE VALUES LARGER THAN 255!!"
    if len(arr[arr[:] < 0]) > 0:
        print "THERE ARE VALUES SMALLER THAN 0!!"
    arr[arr[:] > 255] = 255
    arr[arr[:] < 0] = 0
    return Image.fromarray(np.uint8(arr), mode="L")


def array_info(arr):
    print "***  ARRAY  INFO  ***"
    print "array type: " + str(type(arr))
    print "elements' type: " + str(type(arr[0, 0]))
    print "dimensions: " + str(arr.shape)
    print "mean: " + str(np.mean(arr))
    print "median: " + str(np.median(arr))
    print "min: " + str(np.min(arr))
    print "max: " + str(np.max(arr))
    print "* * * * * * * * * * *"


def psnr(arr1, arr2):
    """
    Peak Signal to Noise Ratio
    :return:
    """
    if arr1.shape != arr2.shape:
        print "WRONG ARRAYS, EXIT"
        return float("inf")
    mse = np.mean((arr1 - arr2) ** 2)
    print "mse: " + str(mse)
    if mse != 0:
        return 10.0 * np.log(65025.0 / mse)
    print "NEW IMAGE IS EQUAL TO THE ORIGINAL, PSNR NOT DEFINED"
    return float("inf")


# Return an array that has zeros around (?) the given array in order to reach the given dimensions
def pad(arr, shape):
    arr2 = np.zeros(shape=shape, dtype=type(arr[0][0]))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr2[i][j] = arr[i][j]
    return arr2


# Clears the padding, and returns the upper left part of the image that should correspond to the original image
def unpad(arr, shape):
    arr2 = np.empty(shape=shape, dtype=type(arr[0][0]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr2[i][j] = arr[i][j]
    return arr2


# Concentric rings
def apply_mask3(mask, fi, fi2):
    h, w = mask.shape
    for i in range(0, h / 2):
        for j in range(0, w):
            if i ** 2 + j ** 2 >= fi ** 2:
                if i ** 2 + j ** 2 <= fi2 ** 2:
                    mask[i][j] = 1
    for i in range(h / 2, h):
        for j in range(0, w):
            if (h - i) ** 2 + j ** 2 >= fi ** 2:
                if (h - i) ** 2 + j ** 2 <= fi2 ** 2:
                    mask[i][j] = 1
    return mask


# arr should be an numpy 2d array
# m: min range, M: max range
def normalize2d(arr, mg):
    m = np.min(arr)
    M = np.max(arr)
    arr2 = arr.copy()
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            arr2[i][j] = ((arr[i][j] - m) / float(M - m)) * mg
    return arr2