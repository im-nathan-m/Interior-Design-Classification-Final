import splitfolders

ROOT_DIR = "archive"

# ------------ Step 0. Data Splitting ------------
def val_split():
    """
    Since our data is already split 0.8/0.2 train/test, we will split the
    training folder 0.875/0.125, so our final dataset split is 0.7/0.1/0.2
    train/val/test.
    """
    splitfolders.ratio(
        input=f'{ROOT_DIR}/dataset_train',
        output=f'{ROOT_DIR}/dataset_train_val',
        seed=1337,
        ratio=(0.875, 0.125),
        group=None
    )

val_split()