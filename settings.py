import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

DICOM_DIR = os.path.join(BASE_DIR, "data", "dicom-images-train")
CSV_PATH = os.path.join(BASE_DIR, "data", "train-rle.csv")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_ROOT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, SCRIPT_NAME)

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_unet_dev_overfit_bad.pth")

# общий режим работы:
# "train" - обучение
# "predict" - только предсказание по уже обученной модели
RUN_MODE = "train"

# режимы обучения:
# "overfit" - проверка, может ли сеть заучить 10 positive снимков
# "balanced" - нормальное обучение на positive + negative
MODE = "balanced"

# источник картинки для предсказания:
# "positive" - случайная положительная
# "negative" - случайная отрицательная
# "random" - случайная любая
PREDICT_SOURCE = "positive"

# параметры overfit
OVERFIT_POSITIVE_COUNT = 10
OVERFIT_EPOCHS = 150 # 50
OVERFIT_IMAGE_SIZE = 128
OVERFIT_BATCH_SIZE = 1 # 2

LEARNING_RATE = 1e-3

# параметры balanced обучения
BALANCED_POSITIVE_COUNT = 2000
BALANCED_NEGATIVE_COUNT = 1200
BALANCED_VAL_RATIO = 0.2
BALANCED_EPOCHS = 50
BALANCED_IMAGE_SIZE = 256
BALANCED_BATCH_SIZE = 2 # 2

# LEARNING_RATE = 1e-4 # todo
SEED = 42
# SEED = 45

# порог для предсказания
PRED_THRESHOLD = 0.2

# если true, после обучения сразу сделает предсказание на одном positive train снимке
RUN_PREDICT_AFTER_TRAIN = True