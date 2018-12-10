import random
import cv2
import numpy as np

class Rotate:
    def __init__(self, limit=10, prob=1):
        self.prob = prob
        self.limit = limit

    def __call__(self, img):

        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width, channel = img.shape
            mat = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        return img


class CropScale:
    def __init__(self, limit=10, prob=1):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            height, width, channel = img.shape
            shift_v = random.choice(range(self.limit))
            shift_h = random.choice(range(self.limit))

            if random.random() < 0.5:
                img = img[shift_v:, shift_h:, :]
            else:
                img = img[:(height - shift_v), :(width - shift_h), :]
            img = cv2.resize(img, (height, width))
        return img


class CropAddBoundary:
    def __init__(self, limit=10, prob=1):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            height, width, channel = img.shape
            shift_v = random.choice(range(self.limit))
            shift_h = random.choice(range(self.limit))

            if random.random() < 0.5:
                img = img[shift_v:, shift_h:, :]
                img = cv2.copyMakeBorder(img, 0, shift_v, 0, shift_h,
                                         borderType=cv2.BORDER_REPLICATE)
            else:
                img = img[:(height - shift_v), :(width - shift_h), :]
                img = cv2.copyMakeBorder(img, shift_v, 0, shift_h, 0,
                                         borderType=cv2.BORDER_REPLICATE)
        return img


class RandomBrightness:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = np.clip(alpha * img[..., :3], 0, maxval).astype(dtype)
        return img


class AddNoise:
    def __init__(self, prob=0.4):
        self.prob = prob

    def __call__(self, image):
        row, col, ch = image.shape
        if random.random() < self.prob:
            if random.random() < 0.5:
              mean = 0
              var = 0.1
              sigma = var**0.5
              gauss = np.random.normal(mean, sigma, (row, col, ch))
              gauss = gauss.reshape(row, col, ch)
              noisy = image + gauss
            else:
              s_vs_p = 0.5
              amount = 0.004
              noisy = np.copy(image)
              # Salt mode
              num_salt = np.ceil(amount * image.size * s_vs_p)
              coords = [np.random.randint(0, i - 1, int(num_salt))
                        for i in image.shape]
              noisy[tuple(coords)] = 1
        else:
            noisy = image
        return noisy


transforms = [Rotate(),
              CropScale(),
              CropAddBoundary(),
              AddNoise(),
              RandomBrightness()]


class MultiCompose:
    def __init__(self, transforms=transforms):
        self.transforms = transforms
        self.num = len(self.transforms)

    def __call__(self, img):
        transform = random.sample(self.transforms, 1)[0]
        img = transform(img)
        return img


multi_transform = MultiCompose()
