# Dataset path
DATASET_DIR = 'dataset/'

# Images paths
NORMAL_IMAGES_FOLDER = 'dataset/Normal/'
TUBERCULOSIS_IMAGES_FOLDER = 'dataset/Tuberculosis'

# Dataframes paths
NORMAL_XLSX_PATH = 'dataset/Normal.metadata.xlsx'
TUBERCULOSIS_XLSX_PATH = 'dataset/Tuberculosis.metadata.xlsx'

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE: float = 0.001
NUM_EPOCHS: int = 10
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 1e-5


DEVICE = 'cuda'
LOGFILE = 'log/model.pt'
TXT_RESULTS = 'log/logs.txt'