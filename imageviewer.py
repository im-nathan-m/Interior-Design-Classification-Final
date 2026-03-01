'''
Documentation consulted:
https://docs.python.org/3/library/os.html
https://docs.python.org/3/library/os.path.html
'''

from PIL import Image
import matplotlib.pyplot as plt
import os


datasets = [
    "dataset_train",
    "dataset_val",
    "dataset_test"
]

styles = [
    "asian",
    "coastal",
    "contemporary",
    "craftsman",
    "eclectic",
    "farmhouse",
    "french-country",
    "industrial",
    "mediterranean",
    "mid-century-modern",
    "modern",
    "rustic",
    "scandinavian",
    "shabby-chic-style",
    "southwestern",
    "traditional",
    "transitional",
    "tropical",
    "victorian"
]

# OPTION 1: view all styles in succession
for style in styles:

# OPTION 2: view only specific style (unindent below code when using)
# set index of desired style
# style = styles[0]

    all_paths = [("archive/" + d + "/" + style) for d in datasets]
    images = []
    labels = []

    for folder in all_paths:
        image_files = [f for f in os.listdir(folder)]

        for file in image_files:
            image_path = os.path.join(folder, file)
            img = Image.open(image_path)
            images.append(img)

            title = file.split('.')[0]  # remove the .jpg file extension
            title = title.replace("mid-century-modern", "mid-century")  # shorten the longest style names to prevent overflow in viewer
            title = title.replace("shabby-chic-style", "shabby-chic")
            labels.append(title)

    num_images = len(images)
    NUM_ROWS = 7
    NUM_COLS = 15
    DISPLAY_MAX = NUM_ROWS * NUM_COLS   # 105

    batch_number = 0

    while (batch_number * DISPLAY_MAX < num_images):
        index = batch_number * DISPLAY_MAX
        image_batch = images[index : index + DISPLAY_MAX]
        label_batch = labels[index : index + DISPLAY_MAX]

        plt.figure(figsize=(16, 8))

        for idx, image in enumerate(image_batch):
            plt1 = plt.subplot(NUM_ROWS, NUM_COLS, idx + 1)
            plt1.imshow(image)
            plt1.set_title(label_batch[idx], fontsize=6)
            plt1.axis('off')

        plt.tight_layout()
        plt.show()

        batch_number += 1