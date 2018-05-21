import numpy as np
from tqdm import tqdm

from src import data_manager, dataset
from src.dataset import get_validation_set


def main():
    validation_set = get_validation_set()
    l1_loss = 0
    for p1, p2, p3 in tqdm(validation_set.tuples):
        p1, p2, p3, = [dataset.pil_to_numpy(data_manager.load_img(p)) for p in (p1, p2, p3)]
        p2_predicted = np.mean([p1, p3], axis=0)
        l1_loss += np.mean(np.abs(p2 - p2_predicted))
    print(f"Linear interpolation: {l1_loss / len(validation_set.tuples)}")


if __name__ == "__main__":
    main()
