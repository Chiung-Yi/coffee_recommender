document.getElementById('getRecommendation').addEventListener('click', function() {
    var userInput = document.getElementById('userInput').value;
    fetch('/recommend/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: 'input=' + encodeURIComponent(userInput)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').textContent = '推薦咖啡：' + data.recommendation;
    });
});

// 獲取 CSRF token 的函數
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