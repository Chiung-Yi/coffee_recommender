from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import chardet
from django.conf import settings
import random

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
                 df['How much caffeine do you like in your coffee?'].fillna('') + ' ' + \
                 df['Gender'].fillna('') + ' ' + \
                 df['Education Level'].fillna('') + ' ' + \
                 df['Ethnicity/Race'].fillna('') + ' ' + \
                 df['Employment Status'].fillna('') + ' ' + \
                 df['Number of Children'].astype(str).fillna('') + ' ' + \
                 df['Political Affiliation'].fillna('')

# 創建CountVectorizer對象
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(df['features'])

def index(request):
    return render(request, 'coffee_recommender.html')

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
    'Regular drip coffee': {'chinese': '普通濾泡咖啡', 'english': 'Regular drip coffee'},
    'Espresso': {'chinese': '義式濃縮咖啡', 'english': 'Espresso'},
    'Latte': {'chinese': '拿鐵咖啡', 'english': 'Latte'},
    'Cappuccino': {'chinese': '卡布奇諾', 'english': 'Cappuccino'},
    'Americano': {'chinese': '美式咖啡', 'english': 'Americano'},
    'Mocha': {'chinese': '摩卡咖啡', 'english': 'Mocha'},
    'Macchiato': {'chinese': '瑪奇朵', 'english': 'Macchiato'},
    'Flat White': {'chinese': '白咖啡', 'english': 'Flat White'},
    'Cortado': {'chinese': '科塔多咖啡', 'english': 'Cortado'},
    'Iced coffee': {'chinese': '冰咖啡', 'english': 'Iced coffee'},
    'Cold brew': {'chinese': '冷萃咖啡', 'english': 'Cold brew'},
    'Pourover': {'chinese': '手沖咖啡', 'english': 'Pourover'},
    'French press': {'chinese': '法式壓濾咖啡', 'english': 'French press'}
}

def recommend_coffee(user_input):
    # 檢查用戶輸入是否為年齡數值或"數值+歲"格式
    age_pattern = r'(\d+)(?:\s*(?:years old|歲))?'
    match = re.match(age_pattern, user_input)

    if match:
        age = int(match.group(1))
        age_group = age_to_group(age)
        user_input = age_group  # 將年齡數值或"數值+歲"轉換為年齡區間
    else:
        # 擴展中英文翻譯字典
        translations = {
            '強': 'strong',
            '弱': 'weak',
            '淡': 'light',
            '濃': 'dark',
            '咖啡因': 'caffeine',
            '牛奶': 'milk',
            '糖': 'sugar',
            '男': 'male',
            '女': 'female',
            '其他性別': 'other gender',
            '高中': 'high school',
            '大學': 'college',
            '研究所': 'graduate school',
            '博士': 'doctorate',
            '全職': 'full-time',
            '兼職': 'part-time',
            '失業': 'unemployed',
            '學生': 'student',
            '退休': 'retired',
            '白人': 'white',
            '黑人': 'black',
            '亞洲人': 'asian',
            '西班牙裔': 'hispanic',
            '已婚': 'married',
            '單身': 'single',
            '有孩子': 'has children',
            '無孩子': 'no children',
            '民主黨': 'democrat',
            '共和黨': 'republican',
            '獨立黨': 'independent',
            '中立': 'neutral'
        }

        # 將用戶輸入中的中文關鍵詞替換為英文
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

    # 如果推薦的咖啡是 nan 或 'Other'，則隨機選擇一種咖啡
    if pd.isna(recommended_coffee) or recommended_coffee == 'Other':
        recommended_coffee = random.choice(list(coffee_translations.keys()))

    # 獲取翻譯後的咖啡名稱和URL
    coffee_data = coffee_translations.get(recommended_coffee, 
                                          {'chinese': recommended_coffee, 'english': recommended_coffee})
    url = f"https://www.catamona1998.com/categories/{recommended_coffee.lower().replace(' ', '-')}"

    return coffee_data, url

def recommend(request):
    if request.method == 'POST':
        user_input = request.POST.get('input', '')
        recommended_coffee, url = recommend_coffee(user_input)
        return JsonResponse({'recommendation': recommended_coffee, 'url': url})
    return JsonResponse({'error': 'Invalid request method'})