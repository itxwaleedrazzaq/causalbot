import cv2
import numpy as np
import pickle

SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 8, 75, 75, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def resize(image):
    """
    Resize the image to the model input size.
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert RGB image to YUV.

    This follows the preprocessing used by the
    NVIDIA/Udacity-style behavioral cloning model.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Apply the complete preprocessing pipeline.

    Input:
        image: RGB image

    Output:
        YUV image with shape:
        (66, 200, 3)
    """
    image = resize(image)
    image = rgb2yuv(image)

    return image


def random_flip(image, command):
    """
    Randomly flip the image horizontally.

    When the image is flipped, the steering angle
    must also be inverted.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

        command = command.copy()
        command[1] = -command[1]

    return image, command

def random_translate(image, command, range_x=100, range_y=10):
    """
    Randomly translate the image horizontally and vertically.

    Horizontal translation also modifies the steering angle.
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)

    # Adjust command according to horizontal translation
    command = command + trans_x * 0.002

    trans_m = np.float32([
        [1, 0, trans_x],
        [0, 1, trans_y]
    ])

    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, command


def random_shadow(image):
    """
    Add a random shadow to the image.
    """
    h, w = image.shape[:2]

    x1 = w * np.random.rand()
    y1 = 0

    x2 = w * np.random.rand()
    y2 = h

    xm, ym = np.mgrid[0:h, 0:w]

    mask = np.zeros_like(image[:, :, 1])

    mask[
        (ym - y1) * (x2 - x1)
        - (y2 - y1) * (xm - x1) > 0
    ] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = (hls[:, :, 1][cond] * s_ratio)

    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly change image brightness.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(image, command, range_x=100, range_y=10):
    """
    Apply random augmentation to image.

    Input:
        image: RGB image
        steering: steering angle

    Output:
        augmented RGB image
        adjusted steering angle
    """
    image, command = random_flip(image, command)
    image, command = random_translate(image, command)
    image = random_shadow(image)
    image = random_brightness(image)

    return image, command


def prepare_data(images, labels, is_training=False):
    processed_images = np.empty(
        (len(images), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        dtype=np.float32
    )
    processed_labels = np.asarray(labels, dtype=np.float32).copy()

    for i in range(len(images)):
        image = np.asarray(images[i])
        command = processed_labels[i].copy()

        if is_training and np.random.rand() < 0.6:
            image, command = augment(image, command)

        image = preprocess(image)
        processed_images[i] = image
        processed_labels[i] = command

    return processed_images, processed_labels


def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    images = data["images"]
    labels = data["labels"]

    print(f"{filename}")
    print(f"  Images: {len(images)}")
    print(f"  Labels: {len(labels)}")

    assert len(images) == len(labels)

    return images, labels


def create_sequences(images, labels, sequence_length):
    num_sequences = len(images) - sequence_length + 1

    X = np.empty(
        (num_sequences, sequence_length, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        dtype=np.float32
    )
    y = np.empty((num_sequences, 2), dtype=np.float32)

    for i in range(num_sequences):
        X[i] = images[i:i + sequence_length]
        y[i] = labels[i + sequence_length - 1]

    return X, y