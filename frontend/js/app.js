// frontend/js/app.js

document.getElementById('file-input').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function () {
            const previewImage = document.getElementById('preview-image');
            previewImage.src = reader.result;
            document.getElementById('preview').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// برای کنترل دکمه‌ها و تعاملات دیگر
document.getElementById('loginBtn').addEventListener('click', () => {
    alert('به‌زودی به صفحهٔ ورود متصل می‌شود');
});

const handleFile = (file) => {
    if (!file) return;
    if (file.size > 15 * 1024 * 1024) {
        alert('حجم فایل بیش از حد مجاز است (حداکثر 15MB).');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImg').src = e.target.result;
        document.getElementById('previewWrap').hidden = false;
        document.getElementById('analyzeBtn').disabled = false;
    };
    reader.readAsDataURL(file);
};

document.getElementById('dropzone').addEventListener('click', () => document.getElementById('fileInput').click());
document.getElementById('fileInput').addEventListener('change', (e) => handleFile(e.target.files[0]));
