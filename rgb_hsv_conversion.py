import numpy as np

def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    "convert (H, W, 3) np.uint8 image with RGB channel to"
    "(H, W, 3) np.float np.ndarray with first channel is hue,"
    "second channel is saturation, and third channel is value."

    normalized = img[:, :, :3]/255.
    HSVmap = np.ndarray((normalized.shape[0], normalized.shape[1], 3), np.float32)
    V = np.max(normalized, axis=2)
    zeroV = np.where(V == 0)
    nonzeroV = np.where(V > 0)
    min = np.min(normalized, axis=2)
    largestmap = np.argmax(normalized, axis=2)
    delta = V - min
    zerodelta = np.where(delta == 0)
    maskR = np.where(largestmap == 0)
    maskG = np.where(largestmap == 1)
    maskB = np.where(largestmap == 2)
    H = np.ndarray(normalized.shape[:2], np.float32)
    H[maskR] = ((normalized[:, :, 1] - normalized[:, :, 2])/delta/6)[maskR]
    H[maskG] = (1/3 + (normalized[:, :, 2] - normalized[:, :, 0])/delta/6)[maskG]
    H[maskB] = (2/3 + (normalized[:, :, 0] - normalized[:, :, 1])/delta/6)[maskB]
    H[zerodelta] = 0
    H %= 1
    S = np.ndarray(normalized.shape[:2], np.float32)
    S[zeroV] = 0
    S[nonzeroV] = (delta/V)[nonzeroV]
    HSVmap[:, :, 0] = H
    HSVmap[:, :, 1] = S
    HSVmap[:, :, 2] = V

    return HSVmap

def hsv_to_rgb(HSVmap:np.ndarray) -> np.ndarray:
    C = HSVmap[:, :, 1]*HSVmap[:, :, 2]
    X = C*(1 - np.abs(6*HSVmap[:, :, 0]%2 - 1))
    m = HSVmap[:, :, 2] - C
    R_ = np.where(HSVmap[:, :, 0] < 1/6, C, 
         np.where(HSVmap[:, :, 0] < 2/6, X,
         np.where(HSVmap[:, :, 0] < 3/6, 0,
         np.where(HSVmap[:, :, 0] < 4/6, 0,
         np.where(HSVmap[:, :, 0] < 5/6, X,
                                         C)))))
    G_ = np.where(HSVmap[:, :, 0] < 1/6, X, 
         np.where(HSVmap[:, :, 0] < 2/6, C,
         np.where(HSVmap[:, :, 0] < 3/6, C,
         np.where(HSVmap[:, :, 0] < 4/6, X,
         np.where(HSVmap[:, :, 0] < 5/6, 0,
                                         0)))))
    B_ = np.where(HSVmap[:, :, 0] < 1/6, 0, 
         np.where(HSVmap[:, :, 0] < 2/6, 0,
         np.where(HSVmap[:, :, 0] < 3/6, X,
         np.where(HSVmap[:, :, 0] < 4/6, C,
         np.where(HSVmap[:, :, 0] < 5/6, C,
                                         X)))))
    R = ((R_ + m)*255).astype(np.uint8)
    G = ((G_ + m)*255).astype(np.uint8)
    B = ((B_ + m)*255).astype(np.uint8)

    rgbimag = np.ndarray(HSVmap.shape, np.uint8)
    rgbimag[:, :, 0] = R
    rgbimag[:, :, 1] = G
    rgbimag[:, :, 2] = B
    return rgbimag