from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 在這裡放置您原有的函數（age_to_group, recommend_coffee 等）

import os
import chardet
from django.conf import settings

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw = file.read()
    result = chardet.detect(raw)
    return result['encoding']

# 使用檢測到的編碼來讀取文件
file_path = os.path.join(settings.BASE_DIR, 'data', 'GACTT_RESULTS_ANONYMIZED_v2.csv')
encoding = detect_encoding(file_path)
df = pd.read_csv(file_path, encoding=encoding)

# 準備數據
df['features'] = df['What is your age?'].astype(str) + ' ' + \
                 df['What kind of dairy do you add?'].fillna('') + ' ' + \
                 df['What kind of sugar or sweetener do you add?'].fillna('') + ' ' + \
                 df['What roast level of coffee do you prefer?'].fillna('') + ' ' + \
                 df['How strong do you like your coffee?'].fillna('') + ' ' + \
                 df['How much caffeine do you like in your coffee?'].fillna('')

# 創建CountVectorizer對象
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(df['features'])

def index(request):
    return render(request, 'recommender/index.html')

def age_to_group(age):
    if age < 18:
        return '<18 years old'
    elif 18 <= age < 25:
        return '18-24 years old'
    elif 25 <= age < 35:
        return '25-34 years old'
    elif 35 <= age < 45:
        return '35-44 years old'
    elif 45 <= age < 55:
        return '45-54 years old'
    elif 55 <= age < 65:
        return '55-64 years old'
    else:
        return '>65 years old'


coffee_translations = {
    'Regular drip coffee': '普通濾泡咖啡',
    'Espresso': '義式濃縮咖啡',
    'Latte': '拿鐵咖啡',
    'Cappuccino': '卡布奇諾',
    'Americano': '美式咖啡',
    'Mocha': '摩卡咖啡',
    'Macchiato': '瑪奇朵',
    'Flat White': '白咖啡',
    'Cortado': '科塔多咖啡',
    'Iced coffee': '冰咖啡',
    'Cold brew': '冷萃咖啡',
    'Pourover': '手沖咖啡',
    'French press': '法式壓濾咖啡'
}

def recommend_coffee(user_input):
    # 檢查用戶輸入是否為年齡數值或"數值+years old"格式
    age_pattern = r'(\d+)(?:\s*(?:years old|歲))?'
    match = re.match(age_pattern, user_input)

    if match:
        age = int(match.group(1))
        age_group = age_to_group(age)
        user_input = age_group  # 將年齡數值或"數值+years old"轉換為年齡區間
    else:
        # 將中文翻譯成英文
        translations = {
            '強': 'strong',
            '弱': 'weak',
            '淡': 'light',
            '濃': 'dark',
            '咖啡因': 'caffeine',
            '牛奶': 'milk',
            '糖': 'sugar'
        }
        for cn, en in translations.items():
            user_input = user_input.replace(cn, en)

    # 將用戶輸入轉換為向量
    user_vector = vectorizer.transform([user_input])

    # 計算餘弦相似度
    cosine_similarities = cosine_similarity(user_vector, feature_matrix)

    # 找到最相似的咖啡
    similar_coffee_index = cosine_similarities.argmax()

    # 獲取推薦的咖啡種類
    recommended_coffee = df.iloc[similar_coffee_index]['What is your favorite coffee drink?']

    # 如果推薦的咖啡是 nan，則推薦美式咖啡
    if pd.isna(recommended_coffee):
        recommended_coffee = 'Americano'

    # 翻譯推薦的咖啡種類
    translated_coffee = coffee_translations.get(recommended_coffee, recommended_coffee)

    return translated_coffee



def recommend(request):
    if request.method == 'POST':
        user_input = request.POST.get('input', '')
        recommended_coffee = recommend_coffee(user_input)
        return JsonResponse({'recommendation': recommended_coffee})
    return JsonResponse({'error': 'Invalid request method'})

from django.apps import apps

def some_view(request):
    df = apps.get_app_config('recommender').df
    # 使用 df 進行操作