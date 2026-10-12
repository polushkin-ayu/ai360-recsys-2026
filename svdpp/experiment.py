"""Полный подбор lr/lambda отдельно для каждого k.

python experiment.py tune --device cuda --k 2
python experiment.py --help
"""
from src.experiment import main

if __name__ == "__main__":
    main()
