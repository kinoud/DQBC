from cv2 import imread as imread_bgr
from skimage.io import imread as imread_rgb


def imread(p,rgb_order):
    if rgb_order=='rgb':
        return imread_rgb(p)
    elif rgb_order=='bgr':
        return imread_bgr(p)
    else:
        raise NotImplementedError
