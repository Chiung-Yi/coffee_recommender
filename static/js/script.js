document.addEventListener('DOMContentLoaded', function() {
    const languageData = {
        'zh-TW': {
            'title': '咖啡推薦系統',
            'description': '寫下關於您自己的一些事：',
            'input-placeholder': '輸入例如:性別、年齡..等',
            'get-recommendation': '獲取推薦',
            'clear': '清除',
            'arabica': '阿拉比卡',
            'arabica-description': '甜美順滑，風味複雜。最受歡迎的咖啡品種。',
            'robusta': '羅布斯塔',
            'robusta-description': '強烈苦味，咖啡因含量高。常用於義式濃縮咖啡',
            'liberica': '利比里卡',
            'liberica-description': '獨特的水果和木質風味。罕見，主要在東南亞種植。',
            'eng': 'ENG',
            'zh': '中文',
            'recommended-coffee': '推薦咖啡：'
        },
        'en': {
            'title': 'Coffee Recommender System',
            'description': 'Write something about yourself:',
            'input-placeholder': 'Input e.g. gender, age, etc.',
            'get-recommendation': 'Get Recommendation',
            'clear': 'Clear',
            'arabica': 'Arabica',
            'arabica-description': 'Sweet and smooth with complex flavors. The most popular coffee variety.',
            'robusta': 'Robusta',
            'robusta-description': 'Strong bitter taste, high caffeine content. Often used in espresso.',
            'liberica': 'Liberica',
            'liberica-description': 'Unique fruity and woody flavor. Rare, mainly grown in Southeast Asia.',
            'eng': 'ENG',
            'zh': '中文',
            'recommended-coffee': 'Recommended Coffee: '
        }
    };

    const languageSwitch = document.getElementById('languageSwitch');
    const getRecommendationButton = document.getElementById('getRecommendation');
    const clearButton = document.getElementById('clearInput');
    const userInput = document.getElementById('userInput');
    const resultDiv = document.getElementById('result');

    let currentLang = 'zh-TW';

    function setLanguage(lang) {
        currentLang = lang;
        document.documentElement.lang = lang;
        const elements = document.querySelectorAll('[data-lang-key]');
        elements.forEach(element => {
            const key = element.getAttribute('data-lang-key');
            if (languageData[lang][key]) {
                if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                    element.placeholder = languageData[lang][key];
                } else {
                    element.textContent = languageData[lang][key];
                }
            }
        });

        if (lang === 'zh-TW') {
            languageSwitch.querySelector('.switch-bar').classList.add('zh');
        } else {
            languageSwitch.querySelector('.switch-bar').classList.remove('zh');
        }
    }

    languageSwitch.addEventListener('click', () => {
        const newLang = currentLang === 'zh-TW' ? 'en' : 'zh-TW';
        setLanguage(newLang);
    });

    getRecommendationButton.addEventListener('click', function() {
        const input = userInput.value;
        fetch('/recommend/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: 'input=' + encodeURIComponent(input)
        })
        .then(response => response.json())
        .then(data => {
            const recommendationText = languageData[currentLang]['recommended-coffee'];
            let coffeeNameText;
            if (currentLang === 'zh-TW') {
                coffeeNameText = data.recommendation.chinese;
            } else {
                coffeeNameText = data.recommendation.english;
            }
            const url = data.recommendation.url;
            resultDiv.innerHTML = `${recommendationText}<a href="${url}" target="_blank">${coffeeNameText}</a>`;
        });
    });

    clearButton.addEventListener('click', function() {
        userInput.value = '';
        resultDiv.textContent = '';
    });

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // 輪播功能
    const carousel = document.querySelector('.coffee-carousel');
    const prevButton = carousel.querySelector('.prev');
    const nextButton = carousel.querySelector('.next');
    const coffeeVarieties = carousel.querySelectorAll('.coffee-variety');
    let currentIndex = 0;

    function showSlide(index) {
        coffeeVarieties.forEach((variety, i) => {
            variety.style.display = i === index ? 'block' : 'none';
        });
    }

    prevButton.addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + coffeeVarieties.length) % coffeeVarieties.length;
        showSlide(currentIndex);
    });

    nextButton.addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % coffeeVarieties.length;
        showSlide(currentIndex);
    });

    // 初始化語言和輪播
    setLanguage(currentLang);
    showSlide(currentIndex);
});
