import kagglehub
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews", output_dir=BASE_DIR / "data" / "raw")
