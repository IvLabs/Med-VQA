class Config:
    LR = 1e-4
    MAX_EPOCHS = 1

    TRAIN_IMG_PATH = "/home/terasquid/Documents/med-VQA/dataset/ImageClef-2019-VQA-Med-Training/resized_images"
    TEST_IMG_PATH = "/home/terasquid/Documents/med-VQA/dataset/ImageClef-2019-VQA-Med-Training/Resize_images_val"

    TRAIN_DATA_DICT_PATH = "/home/terasquid/Documents/med-VQA/dataset/VQA-Med2019/data_dictionary.pkl"
    TEST_DATA_DICT_PATH = "/home/terasquid/Documents/med-VQA/dataset/VQA-Med2019/data_dictionary_val.pkl"

    LOSS_PATH = "home/terasquid/Documents/med-VQA/baselines/Heirarchical/loss_of_epoch1.txt"
    MODEL_STORE_PATH = "home/terasquid/Documents/med-VQA/baselines/Heirarchical/loss_of_epoch1.txt"
    DEVICE = 'cpu'