import Util as utl
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime
import time
import os
from scipy.stats import norm
import math


def dft2(arr):
    return np.fft.rfft2(arr)


def idft2(arr):
    return np.fft.irfft2(arr)


def cart2pol(arr):
    return np.angle(arr), np.absolute(arr)


# returns an numpy array of complex numbers
def pol2complex(phase, magn):
    arr = np.empty(phase.shape, dtype=complex)
    for i in range(0, phase.shape[0]):
        for j in range(0, phase.shape[1]):
            r = magn[i][j]
            theta = phase[i][j]
            arr[i][j] = construct_complex(r * np.sin(theta), r * np.cos(theta))
    return arr


def construct_complex(im, re):
    rev = 1j * im
    rev += re
    return rev


mpad3 = 200
mpad4 = 450


def embed_fourier_watermark(img, watermark, power, shape):
    shape0 = img.shape
    length = len(watermark)
    img = utl.pad(img, shape)

    # DFT
    print "DFT..."
    phase, magn = cart2pol(dft2(img))

    # EMBED WATERMARK on the magn1
    print "Applying masks..."
    mask = np.zeros(shape=magn.shape, dtype=int)
    mask = utl.apply_mask3(mask, mpad3, mpad4)
    # utl.arr2image(utl.normalize2d(mask, 220)).show()
    # utl.arr2image(utl.normalize2d(magn, 100000)).show()
    f = utl.f_from_mask(magn, length, mask)

    print "Embedding watermark..."
    f = utl.embed_watermark_signum(f, watermark, power)
    # print "Detection on f: " + str(utl.detect(f, watermark))

    magn = utl.f_to_mask(f, magn, length, mask)

    # INVERSE DFT
    print "Inverse DFT..."
    img2 = np.absolute(idft2(pol2complex(phase, magn)))
    print "Unpadding ..."
    return utl.unpad(img2, shape0)


def detect_fourier_watermark(img, watermark, shape):
    length = len(watermark)
    img = utl.pad(img, shape)
    # DFT 1
    phase, magn = cart2pol(dft2(img))
    # DETECT WATERMARK on the magn
    mask = np.zeros(shape=magn.shape, dtype=int)
    mask = utl.apply_mask3(mask, mpad3, mpad4)
    f = utl.f_from_mask(magn, length, mask)

    return utl.detect(f, watermark)


def embed(key, length, power, dshape, img1, oim):
    watermark = utl.create_watermark(key, length)
    wimg = embed_fourier_watermark(img1, watermark, power, dshape)
    wimg = utl.unpad(wimg, oim.shape)  # NMZ DEN XREIAZETAI AUTO, GINETAI KAI MESA STO EMBED FOURIER
    # utl.arr2image(wimg).show()
    wimg = utl.normalize2d(wimg, 255)
    wimg = utl.arr2image(wimg)
    return wimg


def plot_save(trueArr, falseArr, name):
    trueMean = np.mean(trueArr)
    trueStd = np.std(trueArr)
    falseMean = np.mean(falseArr)
    falseStd = np.std(falseArr)

    x = np.linspace(np.min([falseMean - 4 * falseStd, trueMean - 4 * trueStd]),
                    np.max([falseMean + 4 * falseStd, trueMean + 4 * trueStd]), 1000)
    plt.plot(x, mlab.normpdf(x, trueMean, trueStd))
    plt.plot(x, mlab.normpdf(x, falseMean, falseStd))

    name += "gauss.png"
    print "Saving file: " + str(name)
    plt.savefig(name)
    # plt.show()
    plt.close()


# the integral minus inf to t under the gaussian curve that is defined by mu, and sigma
def Pfr(t, mu, sigma):
    return norm.cdf((t - mu) / sigma)


# t->t'=2*mu-t
def Pfa(t, mu, sigma):
    return norm.cdf((mu - t) / sigma)


def roc_save(trueArr, falseArr, name):
    trueMean = np.mean(trueArr)
    trueStd = np.std(trueArr)
    falseMean = np.mean(falseArr)
    falseStd = np.std(falseArr)
    x = np.linspace(np.min([falseMean - 4 * falseStd, trueMean - 4 * trueStd]),
                    np.max([falseMean + 4 * falseStd, trueMean + 4 * trueStd]), 1000)
    sigmaTrue = math.sqrt(trueStd)
    sigmaFalse = math.sqrt(falseStd)

    pfr = Pfr(x, trueMean, sigmaTrue)
    pfa = Pfa(x, falseMean, sigmaFalse)

    r = (pfa[:] > 0) & (pfr[:] > 0)  # & (pfa[:] < 1) & (pfr[:] < 1)
    pfr = pfr[r]
    pfa = pfa[r]
    x = x[r]

    plt.plot(pfa, pfr)
    plt.plot(x, x)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([10e-200, 10.])
    plt.xlim([10e-200, 10.])
    plt.grid(True)

    name += "roc.png"
    print "Saving file: " + str(name)
    plt.savefig(name)
    # plt.show()
    plt.close()


def im_noise(wimg1, wimg2, wimg3, magn):
    return (utl.image_noise(wimg1, magnitude=magn),
            utl.image_noise(wimg2, magnitude=magn),
            utl.image_noise(wimg3, magnitude=magn))


def im_crop(wimg1, wimg2, wimg3, pd):
    return (utl.image_cropping(wimg1, pd),
            utl.image_cropping(wimg2, pd),
            utl.image_cropping(wimg3, pd))


def im_rotate(wimg1, wimg2, wimg3, dg):
    return (utl.image_rotate(wimg1, dg),
            utl.image_rotate(wimg2, dg),
            utl.image_rotate(wimg3, dg))


def im_resize(wimg1, wimg2, wimg3, sc):
    return (utl.image_resize(wimg1, sc),
            utl.image_resize(wimg2, sc),
            utl.image_resize(wimg3, sc))


def im_mean(wimg1, wimg2, wimg3):
    return (utl.image_blur(wimg1),
            utl.image_blur(wimg2),
            utl.image_blur(wimg3))


def im_median(wimg1, wimg2, wimg3, window):
    return (utl.image_median(wimg1, window),
            utl.image_median(wimg2, window),
            utl.image_median(wimg3, window))


fname1 = "images/gbear.tif"
fname2 = "images/castle.jpg"
fname3 = "images/gf16.tif"
fname4 = "images/gpills.tif"
fname5 = "images/gpeppers.tif"

img1 = cv2.imread(fname1, 0)
img2 = cv2.imread(fname3, 0)
img3 = cv2.imread(fname4, 0)

repetitions = 2
noiseMagnitude = 50
power = 3000
dshape = (1000, 1000)

originalImg1 = img1.copy()
originalImg2 = img2.copy()
originalImg3 = img3.copy()

length = 100  # img1.size /2
key = int(time.time())
message = [1]

trueD1 = []
trueD2 = []
trueD3 = []

falseD1 = []
falseD2 = []
falseD3 = []

psnr1 = []
psnr2 = []
psnr3 = []

start = time.time()

notes = "test"

for i in range(0, repetitions):
    print "LOOP: " + str(i)
    key += i
    wimg1 = embed(key, length, power, dshape, img1, originalImg1)
    wimg2 = embed(key, length, power, dshape, img2, originalImg2)
    wimg3 = embed(key, length, power, dshape, img3, originalImg3)

    # ATTACKS
    # wimg1, wimg2, wimg3 = im_noise(wimg1, wimg2, wimg3, noiseMagnitude)
    # wimg1, wimg2, wimg3 = im_resize(wimg1, wimg2, wimg3, .99)  # no
    # wimg1, wimg2, wimg3 = im_crop(wimg1, wimg2, wimg3, .6)  # ok
    # wimg1, wimg2, wimg3 = im_rotate(wimg1, wimg2, wimg3, -1) # no
    # wimg1, wimg2, wimg3 = im_mean(wimg1, wimg2, wimg3) #
    # wimg1, wimg2, wimg3 = im_median(wimg1, wimg2, wimg3, 3) #


    wimg1.show()
    wimg2.show()
    wimg3.show()
    #
    # wimg1.save("bearNoise50.jpg")
    # wimg2.save("f16Noise50.jpg")
    # wimg3.save("pillsNoise50.jpg")

    wimg1 = np.asarray(wimg1)
    wimg2 = np.asarray(wimg2)
    wimg3 = np.asarray(wimg3)
    d = detect_fourier_watermark(wimg1, utl.create_watermark(key, length), dshape)
    trueD1.append(d)
    d = detect_fourier_watermark(wimg2, utl.create_watermark(key, length), dshape)
    trueD2.append(d)
    d = detect_fourier_watermark(wimg3, utl.create_watermark(key, length), dshape)
    trueD3.append(d)

    psnr1.append(utl.psnr(originalImg1, wimg1))
    psnr2.append(utl.psnr(originalImg2, wimg2))
    psnr3.append(utl.psnr(originalImg3, wimg3))

    d2 = detect_fourier_watermark(wimg1, utl.create_watermark(key + 1, length), dshape)
    falseD1.append(d2)
    d2 = detect_fourier_watermark(wimg2, utl.create_watermark(key + 1, length), dshape)
    falseD2.append(d2)
    d2 = detect_fourier_watermark(wimg3, utl.create_watermark(key + 1, length), dshape)
    falseD3.append(d2)
    # utl.arr2image(utl.normalize2d(mask, 100)).show()

print "Passed: " + str(time.time() - start) + " SECONDS"

print "1. True mean, std: " + str(np.mean(trueD1)) + " , " + str(np.std(trueD1))
print "2. True mean, std: " + str(np.mean(trueD2)) + " , " + str(np.std(trueD2))
print "3. True mean, std: " + str(np.mean(trueD3)) + " , " + str(np.std(trueD3))

print "1. False mean, std: " + str(np.mean(falseD1)) + " , " + str(np.std(falseD1))
print "2. False mean, std: " + str(np.mean(falseD2)) + " , " + str(np.std(falseD2))
print "3. False mean, std: " + str(np.mean(falseD3)) + " , " + str(np.std(falseD3))

print "1. Avg psnr1: " + str(np.mean(psnr1))
print "2. Avg psnr2: " + str(np.mean(psnr2))
print "3. Avg psnr3: " + str(np.mean(psnr3))

folder_name = str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M")) + str(notes)
# create folder
os.makedirs(folder_name)

plot_save(trueD1, falseD1, str(folder_name) + "\\one")
plot_save(trueD2, falseD2, str(folder_name) + "\\two")
plot_save(trueD3, falseD3, str(folder_name) + "\\three")
plot_save(trueD1 + trueD2 + trueD3, falseD1 + falseD2 + falseD3, str(folder_name) + "\\total")

roc_save(trueD1, falseD1, str(folder_name) + "\\one")
roc_save(trueD2, falseD2, str(folder_name) + "\\two")
roc_save(trueD3, falseD3, str(folder_name) + "\\three")
roc_save(trueD1 + trueD2 + trueD3, falseD1 + falseD2 + falseD3, str(folder_name) + "\\total")


file = open(str(folder_name) + "\\details.csv", "w")
file.write("time,psnr1,psnr2,psnr3,trueMean,trueStd,falseMean,falseStd,repetitions,power,notes\n")
file.write(
    str(time.time() - start) + "," + str(np.mean(psnr1)) + "," + str(np.mean(psnr2)) + "," + str(np.mean(psnr3)) + ",")
file.write(str(np.mean(trueD1 + trueD2 + trueD3)) + "," + str(np.std(trueD1 + trueD2 + trueD3)) + ",")
file.write(str(np.mean(falseD1 + falseD2 + falseD3)) + "," + str(np.std(falseD1 + falseD2 + falseD3)) + ",")
file.write(str(repetitions) + "," + str(power) + "," + str(notes))

file.close()
