from django.apps import AppConfig
import pandas as pd
import os
from django.conf import settings
import chardet
import csv

class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'
    df = None

    def detect_encoding(self, file_path):
        with open(file_path, 'rb') as file:
            raw = file.read()
        result = chardet.detect(raw)
        return result['encoding']

    def ready(self):
    try:
        file_path = os.path.join(settings.BASE_DIR, 'data', 'GACTT_RESULTS_ANONYMIZED_v2.csv')
        encoding = self.detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding) as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        
        # 假設第一行是標題
        self.df = pd.DataFrame(data[1:], columns=data[0])
        
        print(f"Successfully read CSV file. Shape: {self.df.shape}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        self.df = pd.DataFrame()  # 創建一個空的 DataFrame