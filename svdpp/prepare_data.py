"""Скачать полный ML1M или проверить уже включённые в архив данные.

python prepare_data.py
python prepare_data.py --check-only
"""
from src.prepare_movielens1m import main

if __name__ == "__main__":
    main()
