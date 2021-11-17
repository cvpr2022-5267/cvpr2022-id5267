import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


class Augmenter(object):
    def __init__(self):
        # set global seed
        ia.seed(53)

        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image.
        self.seq = iaa.Sequential(
            [
                #
                # Apply the following augmenters to most images.

                # crop some of the images by 0-10% of their height/width
                self.sometimes(iaa.CropAndPad(percent=(0, 0.05), pad_mode=ia.ALL, pad_cval=(0, 255))),

                self.sometimes(iaa.GammaContrast((0.5, 2.0))),

                self.sometimes(iaa.AddToBrightness((-30, 30))),

                self.sometimes(iaa.AverageBlur(k=(1, 3))),

                # Apply affine transformations to some of the images
                # - scale to 80-120% of image height/width (each axis independently)
                # - translate by -20 to +20 relative to height/width (per axis)
                # - order: use nearest neighbour or bilinear interpolation (fast)
                # - mode: use any available mode to fill newly created pixels
                #         see API or scikit-image for which modes are available
                # - cval: if the mode is constant, then use a random brightness
                #         for the newly created pixels (e.g. sometimes black,
                #         sometimes white)
                self.sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                self.sometimes(
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.00, 0.05), size_percent=(0.02, 0.25),
                            per_channel=0.2
                        ),
                    ]),
                ),

            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def generate(self, images, segmaps=None):
        t, n, h, w, c = images.shape
        images = images.reshape(t * n, h, w, c)
        if segmaps is None:
            images = self.seq(images=(images*255).astype(np.uint8))
        else:
            segmaps = segmaps.reshape(t * n, h, w, 1)
            images, segmaps = self.seq(images=(images*255).astype(np.uint8), segmentation_maps=segmaps)
            segmaps = segmaps.reshape(t, n, h, w, 1)

        images = images.reshape(t, n, h, w, c)
        if segmaps is None:
            return (images/255.0).astype(np.float32)
        else:
            return (images/255.0).astype(np.float32), segmaps
