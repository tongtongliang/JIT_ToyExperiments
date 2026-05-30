from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_jax_exp.train_representation_ufcn import main

if __name__ == "__main__":
    main()
