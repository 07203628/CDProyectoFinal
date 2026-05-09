import shutil
from pathlib import Path

import kagglehub


DATASET_ID = 'stevezhenghp/airbnb-price-prediction'
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'Datos'
RAW_PATH = DATA_DIR / 'raw.csv'


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    train_file = dataset_path / 'train.csv'

    if not train_file.exists():
        raise FileNotFoundError(f'No se encontró train.csv dentro de {dataset_path}')

    shutil.copy2(train_file, RAW_PATH)
    print(f'Dataset descargado y copiado en: {RAW_PATH}')


if __name__ == '__main__':
    main()
