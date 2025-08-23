// backend/server.js
const express = require('express');
const path = require('path');
const app = express();

// تنظیمات مسیرهای ثابت
app.use(express.static(path.join(__dirname, '../frontend')));

// روت برای صفحه اصلی
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/views/index.html'));
});

// تنظیم پورت
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
