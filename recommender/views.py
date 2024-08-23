from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pandas.api.types import is_numeric_dtype
import re
import os
import chardet
from django.conf import settings
import random
from django.views.decorators.csrf import csrf_exempt

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw = file.read()
    result = chardet.detect(raw)
    return result['encoding']

# 使用檢測到的編碼來讀取文件
file_path = os.path.join(settings.BASE_DIR, 'data', 'GACTT_RESULTS_ANONYMIZED_v2.csv')
encoding = detect_encoding(file_path)
df = pd.read_csv(file_path, encoding=encoding)
print(f"Successfully read CSV file. Shape: {df.shape}")

# 數據預處理
df.ffill(axis=0, inplace=True)

# 選擇用於聚類的特徵
cluster_columns = [
    'What is your age?',
    'What is your favorite coffee drink?',
    'What kind of dairy do you add?',
    'What kind of sugar or sweetener do you add?',
    'What roast level of coffee do you prefer?',
    'Gender',
    'Ethnicity/Race',
    'Education Level',
    'Employment Status',
    'Number of Children'
]

df_clus = df[cluster_columns].copy()

def convert_children_to_numeric(value):
    if pd.isna(value):
        return 0
    elif value == 'More than 3':
        return 4  # 或其他適當的數值
    else:
        try:
            return int(value)
        except ValueError:
            return 0  # 如果無法轉換為整數，返回0

# 應用轉換函數
df_clus['Number of Children'] = df_clus['Number of Children'].apply(convert_children_to_numeric)

# 創建 ColumnTransformer 以處理混合類型的數據
numeric_features = ['Number of Children']
categorical_features = [col for col in cluster_columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 擬合預處理器並轉換數據
X = preprocessor.fit_transform(df_clus)

# 執行 K-means 聚類
kmeans = KMeans(n_clusters=13, random_state=123)
kmeans.fit(X)
df_clus['cluster'] = kmeans.labels_

def index(request):
    return render(request, 'coffee_recommender.html')

coffee_translations = {
    'Regular drip coffee': {'chinese': '普通濾泡咖啡', 'english': 'Regular drip coffee', 'url': 'https://www.catamona1998.com/products/catamona-south-america-100-ground-coffee-1'},
    'Espresso': {'chinese': '義式濃縮咖啡', 'english': 'Espresso', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-coffee-beans'},
    'Latte': {'chinese': '拿鐵咖啡', 'english': 'Latte', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-coffee-beans'},
    'Cappuccino': {'chinese': '卡布奇諾', 'english': 'Cappuccino', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-coffee-beans'},
    'Americano': {'chinese': '美式咖啡', 'english': 'Americano', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-colombia'},
    'Mocha': {'chinese': '摩卡咖啡', 'english': 'Mocha', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-colombia'},
    'Macchiato': {'chinese': '瑪奇朵', 'english': 'Macchiato', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-colombia'},
    'Flat White': {'chinese': '白咖啡', 'english': 'Flat White', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-colombia'},
    'Cortado': {'chinese': '科塔多咖啡', 'english': 'Cortado', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-colombia'},
    'Iced coffee': {'chinese': '冰咖啡', 'english': 'Iced coffee', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-colombia'},
    'Cold brew': {'chinese': '冷萃咖啡', 'english': 'Cold brew', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-colombia'},
    'Pourover': {'chinese': '手沖咖啡', 'english': 'Pourover', 'url': 'https://www.catamona1998.com/products/ciouwaalishancoffeebeans'},
    'French press': {'chinese': '法式壓濾咖啡', 'english': 'French press', 'url': 'https://www.catamona1998.com/products/rainforest-alliance-colombia'}
}

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

def translate_input(user_input):
    # 將用戶輸入中的中文關鍵詞替換為英文
    for cn, en in translations.items():
        user_input = user_input.replace(cn, en)
    return user_input

def parse_user_input(user_input):
    parsed_input = {col: np.nan for col in cluster_columns}  # 使用 np.nan 替代 'Unknown'
    
    # 先將輸入翻譯成英文
    user_input = translate_input(user_input)
    
    # 解析年齡
    age_match = re.search(r'\d+', user_input)
    if age_match:
        age = int(age_match.group())
        parsed_input['What is your age?'] = age_to_group(age)
    
    # 解析咖啡偏好
    for coffee, data in coffee_translations.items():
        if coffee.lower() in user_input.lower() or data['chinese'] in user_input:
            parsed_input['What is your favorite coffee drink?'] = coffee
            break
    
    # 解析其他特徵
    if 'male' in user_input.lower():
        parsed_input['Gender'] = 'Male'
    elif 'female' in user_input.lower():
        parsed_input['Gender'] = 'Female'
    
    if 'milk' in user_input.lower():
        parsed_input['What kind of dairy do you add?'] = 'Milk'
    elif 'cream' in user_input.lower():
        parsed_input['What kind of dairy do you add?'] = 'Cream'
    
    if 'sugar' in user_input.lower():
        parsed_input['What kind of sugar or sweetener do you add?'] = 'Sugar'
    elif 'sweetener' in user_input.lower():
        parsed_input['What kind of sugar or sweetener do you add?'] = 'Artificial sweetener'
    
    if 'light' in user_input.lower():
        parsed_input['What roast level of coffee do you prefer?'] = 'Light roast'
    elif 'medium' in user_input.lower():
        parsed_input['What roast level of coffee do you prefer?'] = 'Medium roast'
    elif 'dark' in user_input.lower():
        parsed_input['What roast level of coffee do you prefer?'] = 'Dark roast'
    
    # 這裡可以繼續添加更多特徵的解析邏輯
    
    return parsed_input

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

def recommend_coffee(user_input):
    # 解析用戶輸入
    parsed_input = parse_user_input(user_input)
    
    # 將 np.nan 替換為一個合適的默認值
    for col in cluster_columns:
        if pd.isna(parsed_input[col]):
            if col == 'Number of Children':
                parsed_input[col] = 0  # 或者其他適當的默認值
            else:
                parsed_input[col] = 'Unknown'  # 對於分類特徵

    # 創建 DataFrame 並處理缺失值
    user_df = pd.DataFrame([parsed_input])
    user_df['Number of Children'] = user_df['Number of Children'].fillna(0)
    
    # 使用預處理器轉換用戶輸入
    user_features = preprocessor.transform(user_df)
    
    # 預測用戶所屬的聚類
    user_cluster = kmeans.predict(user_features)[0]

    # 從相同聚類中選擇最相似的咖啡
    cluster_data = df_clus[df_clus['cluster'] == user_cluster]
    
    # 使用餘弦相似度找到最相似的咖啡
    cosine_similarities = cosine_similarity(user_features, preprocessor.transform(cluster_data[cluster_columns]))
    similar_coffee_index = cosine_similarities.argmax()

    # 獲取推薦的咖啡種類
    recommended_coffee = cluster_data.iloc[similar_coffee_index]['What is your favorite coffee drink?']

    # 如果推薦的咖啡是 nan 或 'Other'，則隨機選擇一種咖啡
    if pd.isna(recommended_coffee) or recommended_coffee == 'Other':
        recommended_coffee = random.choice(list(coffee_translations.keys()))

    # 獲取翻譯後的咖啡名稱和URL
    coffee_data = coffee_translations.get(recommended_coffee, 
                                          {'chinese': recommended_coffee, 'english': recommended_coffee, 'url': ''})

    return coffee_data

@csrf_exempt
def recommend(request):
    if request.method == 'POST':
        user_input = request.POST.get('input', '')
        recommended_coffee = recommend_coffee(user_input)
        return JsonResponse({'recommendation': recommended_coffee})
    return JsonResponse({'error': 'Invalid request method'})