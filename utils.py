import numpy as np

def random_crop(image, crop_size):
    height, width = image.shape[1:]
    dy, dx = crop_size
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return image[:, y:(y+dy), x:(x+dx)]