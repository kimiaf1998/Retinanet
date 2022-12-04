import numpy as np
import torch
import torch.nn as nn

from net.utility.constants import PYRAMID_LEVELS, RATIOS, SCALES


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        if ratios is None:
            self.ratios = np.array(RATIOS)
        if scales is None:
            self.scales = np.array(SCALES)
        if pyramid_levels is None:
            self.pyramid_levels = PYRAMID_LEVELS
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]

    def forward(self, image):

        shape = image.shape[2:]
        shape = np.array(shape)
        shapes = [(shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            _anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(shapes[idx], self.strides[idx], _anchors)
            anchors = np.append(anchors, shifted_anchors, axis=0)

        anchors = np.expand_dims(anchors, axis=0)

        to_return = torch.from_numpy(anchors.astype(np.float32))
        return to_return.cuda() if torch.cuda.is_available() else to_return
    

def generate_anchors(base_size=16, ratios=np.array([0.5, 1, 2]),
                     scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])):
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def compute_shape(image_shape, pyramid_levels):
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(image_shape, pyramid_levels=None, ratios=None, scales=None, strides=None, sizes=None,
                      shapes_callback=None,
                      ):
    image_shapes = compute_shape(image_shape, pyramid_levels)

    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    """ add A anchors (1, A, 4) to
    cell K shifts (K, 1, 4) to get
    shift anchors (K, A, 4)
    reshape to (K*A, 4) shifted anchors"""

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors