import numpy as np

SAMPLING_INTERVAL = 4
UPSCALE = 4


def FourSimplexInterp(weight, img_in, h, w, q, rot, upscale=4):
    L = 2 ** (8 - SAMPLING_INTERVAL) + 1

    # Extract MSBs
    img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
    img_b1 = img_in[:, 0:0 + h, 1:1 + w] // q
    img_c1 = img_in[:, 1:1 + h, 0:0 + w] // q
    img_d1 = img_in[:, 1:1 + h, 1:1 + w] // q

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    # Extract LSBs
    fa_ = img_in[:, 0:0 + h, 0:0 + w] % q
    fb_ = img_in[:, 0:0 + h, 1:1 + w] % q
    fc_ = img_in[:, 1:1 + h, 0:0 + w] % q
    fd_ = img_in[:, 1:1 + h, 1:1 + w] % q

    # Vertices (O in Eq3 and Table3 in the paper)
    p0000 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0001 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    # Naive pixelwise output value interpolation (Table3 in the paper)
    # It would be faster implemented with a parallel operation
    for c in range(img_a1.shape[0]):
        for y in range(img_a1.shape[1]):
            for x in range(img_a1.shape[2]):
                fa = fa_[c, y, x]
                fb = fb_[c, y, x]
                fc = fc_[c, y, x]
                fd = fd_[c, y, x]

                if fa > fb:
                    if fb > fc:
                        if fc > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fb) * p1000[c, y, x] + (fb - fc) * p1100[
                                c, y, x] + (fc - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fb > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fb) * p1000[c, y, x] + (fb - fd) * p1100[
                                c, y, x] + (fd - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                        elif fa > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fd) * p1000[c, y, x] + (fd - fb) * p1001[
                                c, y, x] + (fb - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fa) * p0001[c, y, x] + (fa - fb) * p1001[
                                c, y, x] + (fb - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                    elif fa > fc:
                        if fb > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fc) * p1000[c, y, x] + (fc - fb) * p1010[
                                c, y, x] + (fb - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fc > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fc) * p1000[c, y, x] + (fc - fd) * p1010[
                                c, y, x] + (fd - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                        elif fa > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fd) * p1000[c, y, x] + (fd - fc) * p1001[
                                c, y, x] + (fc - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fa) * p0001[c, y, x] + (fa - fc) * p1001[
                                c, y, x] + (fc - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                    else:
                        if fb > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fa) * p0010[c, y, x] + (fa - fb) * p1010[
                                c, y, x] + (fb - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fc > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fa) * p0010[c, y, x] + (fa - fd) * p1010[
                                c, y, x] + (fd - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                        elif fa > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fd) * p0010[c, y, x] + (fd - fa) * p0011[
                                c, y, x] + (fa - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fc) * p0001[c, y, x] + (fc - fa) * p0011[
                                c, y, x] + (fa - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]

                else:
                    if fa > fc:
                        if fc > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fa) * p0100[c, y, x] + (fa - fc) * p1100[
                                c, y, x] + (fc - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fa > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fa) * p0100[c, y, x] + (fa - fd) * p1100[
                                c, y, x] + (fd - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                        elif fb > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fd) * p0100[c, y, x] + (fd - fa) * p0101[
                                c, y, x] + (fa - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fb) * p0001[c, y, x] + (fb - fa) * p0101[
                                c, y, x] + (fa - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                    elif fb > fc:
                        if fa > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fc) * p0100[c, y, x] + (fc - fa) * p0110[
                                c, y, x] + (fa - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fc > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fc) * p0100[c, y, x] + (fc - fd) * p0110[
                                c, y, x] + (fd - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                        elif fb > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fd) * p0100[c, y, x] + (fd - fc) * p0101[
                                c, y, x] + (fc - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fb) * p0001[c, y, x] + (fb - fc) * p0101[
                                c, y, x] + (fc - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                    else:
                        if fa > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fb) * p0010[c, y, x] + (fb - fa) * p0110[
                                c, y, x] + (fa - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fb > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fb) * p0010[c, y, x] + (fb - fd) * p0110[
                                c, y, x] + (fd - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                        elif fc > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fd) * p0010[c, y, x] + (fd - fb) * p0011[
                                c, y, x] + (fb - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fc) * p0001[c, y, x] + (fc - fb) * p0011[
                                c, y, x] + (fb - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]

    out = np.transpose(out, (0, 1, 3, 2, 4)).reshape(
        (img_a1.shape[0], img_a1.shape[1] * upscale, img_a1.shape[2] * upscale))
    out = np.rot90(out, rot, [1, 2])
    out = out / q
    return out


def sr(img, LUT_PATH):
    # load lut
    LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE * UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)

    # Load LR image
    img_lr = img.astype(np.float32)
    h, w, c = img_lr.shape

    # Sampling interval for input
    q = 2 ** SAMPLING_INTERVAL

    # Rotational ensemble
    img_in = np.pad(img_lr, ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r0 = FourSimplexInterp(LUT, img_in, h, w, q, 0, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 1), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r1 = FourSimplexInterp(LUT, img_in, w, h, q, 3, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 2), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r2 = FourSimplexInterp(LUT, img_in, h, w, q, 2, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 3), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r3 = FourSimplexInterp(LUT, img_in, w, h, q, 1, upscale=UPSCALE)

    img_out = (out_r0/1.0 + out_r1/1.0 + out_r2/1.0 + out_r3/1.0) / 255.0
    img_out = img_out.transpose((1,2,0))
    img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)

    return img_out[..., ::-1]
