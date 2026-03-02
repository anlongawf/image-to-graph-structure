document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const removeBtn = document.getElementById('remove-btn');
    const processBtn = document.getElementById('process-btn');
    const statusLog = document.getElementById('status-log');
    const spinner = document.getElementById('spinner');

    // Result elements
    const textOutput = document.getElementById('text-output');
    const jsonOutput = document.getElementById('json-output');
    const visualResultImg = document.getElementById('visual-result');
    const resultTabText = document.getElementById('tab-text');
    const resultTabJson = document.getElementById('tab-json');
    const resultTabVisual = document.getElementById('tab-visual');

    const contentText = document.getElementById('content-text');
    const contentJson = document.getElementById('content-json');
    const contentVisual = document.getElementById('content-visual');

    const downloadJsonBtn = document.getElementById('download-json');
    const shareBtn = document.getElementById('share-btn');

    // Auth elements
    const authModal = document.getElementById('auth-modal');
    const authTitle = document.getElementById('auth-title');
    const authEmail = document.getElementById('auth-email');
    const authPassword = document.getElementById('auth-password');
    const authSubmitBtn = document.getElementById('auth-submit-btn');
    const authToggle = document.getElementById('auth-toggle');
    const authError = document.getElementById('auth-error');
    const closeBtn = document.querySelector('.close-modal');

    const btnLogin = document.getElementById('btn-login');
    const btnLogout = document.getElementById('btn-logout');
    const btnHistory = document.getElementById('btn-history');
    const userGreeting = document.getElementById('user-greeting');

    let currentFile = null;
    let resultData = null;
    let isRegisterMode = false;

    // Check Auth on load
    function checkAuth() {
        const token = localStorage.getItem('token');
        if (token) {
            btnLogin.style.display = 'none';
            btnLogout.style.display = 'inline-block';
            btnHistory.style.display = 'inline-block';
            // Simple username parse (for demo, real app needs /me endpoint)
            const payloadDecoded = JSON.parse(atob(token.split('.')[1]));
            userGreeting.textContent = `User ID: ${payloadDecoded.sub}`;
            userGreeting.style.display = 'inline-block';
        } else {
            btnLogin.style.display = 'inline-block';
            btnLogout.style.display = 'none';
            btnHistory.style.display = 'none';
            userGreeting.style.display = 'none';
        }
    }

    // Auth Modal Logic
    btnLogin.addEventListener('click', () => { authModal.style.display = 'block'; });
    closeBtn.addEventListener('click', () => { authModal.style.display = 'none'; });
    window.addEventListener('click', (e) => { if (e.target == authModal) authModal.style.display = 'none'; });

    authToggle.addEventListener('click', (e) => {
        e.preventDefault();
        isRegisterMode = !isRegisterMode;
        authTitle.textContent = isRegisterMode ? 'Đăng Ký' : 'Đăng Nhập';
        authSubmitBtn.textContent = isRegisterMode ? 'Đăng Ký' : 'Đăng Nhập';
        authToggle.textContent = isRegisterMode ? 'Đã có tài khoản? Đăng nhập' : 'Chưa có tài khoản? Đăng ký ngay';
        authError.textContent = '';
    });

    authSubmitBtn.addEventListener('click', async () => {
        const email = authEmail.value;
        const password = authPassword.value;
        if (!email || !password) return authError.textContent = "Vui lòng nhập đủ thông tin";

        try {
            const endpoint = isRegisterMode ? '/api/v1/auth/register' : '/api/v1/auth/login';
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail);

            localStorage.setItem('token', data.access_token);
            authModal.style.display = 'none';
            checkAuth();
            log(`Đã đăng nhập thành công.`, 'success');
        } catch (err) {
            authError.textContent = err.message;
        }
    });

    btnLogout.addEventListener('click', () => {
        localStorage.removeItem('token');
        checkAuth();
        log('Đã đăng xuất.');
    });

    function log(message, type = "info", skipTimeFormat = false) {
        let textContent = message;
        if (!skipTimeFormat) {
            const time = new Date().toLocaleTimeString();
            textContent = `[${time}] ${message}`;
        }

        const div = document.createElement('div');
        div.textContent = textContent;
        if (type === 'error') div.style.color = 'var(--error-color)';
        if (type === 'success') div.style.color = 'var(--success-color)';
        if (skipTimeFormat) div.style.color = '#888'; // Custom color for algorithm logs

        statusLog.appendChild(div);
        statusLog.scrollTop = statusLog.scrollHeight;
    }

    // Drag and drop events
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    removeBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        uploadArea.style.display = 'block';
        processBtn.disabled = true;
        processBtn.textContent = 'Chạy Thuật Toán Nhận Diện';
        log('Đã xóa ảnh.');
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            log('Vui lòng chọn file hình ảnh hợp lệ (.png, .jpg)', 'error');
            return;
        }
        if (file.size > 5 * 1024 * 1024) {
            log('File quá lớn. Vui lòng chọn ảnh < 5MB', 'error');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadArea.style.display = 'none';
            previewContainer.style.display = 'flex';
            processBtn.disabled = false;
            processBtn.textContent = 'Chạy Thuật Toán Nhận Diện';
            log(`Đã tải ảnh: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
        };
        reader.readAsDataURL(file);
    }

    // Process Image
    processBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const formData = new FormData();
        formData.append('file', currentFile);

        log('Đang gửi ảnh lên máy chủ...');
        processBtn.disabled = true;
        spinner.style.display = 'block';
        contentText.classList.remove('active');
        contentJson.classList.remove('active');
        contentVisual.classList.remove('active');

        // Prepare Token
        const token = localStorage.getItem('token');
        if (!token) {
            log('Bạn cần ĐĂNG NHẬP để chạy thuật toán.', 'error');
            authModal.style.display = 'block';
            processBtn.disabled = false;
            spinner.style.display = 'none';
            return;
        }

        try {
            const response = await fetch('/api/v1/extract/', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Lỗi không xác định từ máy chủ');
            }

            log(`Xử lý thành công trong ${data.execution_time_ms.toFixed(0)} ms`, 'success');
            renderResult(data.data, data.result_image_path, data.id);

        } catch (error) {
            log(`Lỗi: ${error.message}`, 'error');
        } finally {
            processBtn.disabled = false;
            spinner.style.display = 'none';
        }
    });

    // --- Render logic extracted for reuse ---
    function renderResult(data, imgPath, historyId) {
        resultData = data;

        // Hide upload area, show preview
        document.getElementById('upload-area').style.display = 'none';
        previewContainer.style.display = 'flex';
        // Note: original preview Image is not set here if loaded from history

        const nodes = resultData.nodes || [];
        const edges = resultData.edges || [];

        let htmlContent = `<h3>1. Tổng quan đỉnh (Nodes)</h3>`;
        htmlContent += `<p>Hệ thống tìm thấy <b>${nodes.length}</b> đỉnh trong sơ đồ, bao gồm: <b>${nodes.map(n => n.label).join(', ')}</b>.</p>`;

        htmlContent += `<h3>2. Mối liên kết (Edges)</h3>`;
        if (edges.length === 0) {
            htmlContent += `<p>Không tìm thấy đường nối nào giữa các đỉnh.</p>`;
        } else {
            htmlContent += `<ul>`;
            edges.forEach(edge => {
                const weightText = edge.weight ? ` (Trọng số: ${edge.weight})` : '';
                htmlContent += `<li>Đỉnh <b>${edge.from}</b> nối tới đỉnh <b>${edge.to}</b>${weightText}</li>`;
            });
            htmlContent += `</ul>`;
        }
        textOutput.innerHTML = htmlContent;

        // Render JSON
        const objForJson = { ...resultData };
        delete objForJson.logs;
        jsonOutput.textContent = JSON.stringify(objForJson, null, 2);

        // Render Image
        visualResultImg.src = imgPath;

        // Render Logs to System Log Interface
        statusLog.innerHTML = ''; // Reset all previous logs to make view 100% identical
        if (resultData.logs && resultData.logs.length > 0) {
            resultData.logs.forEach(logLine => {
                log(logLine, 'info', true);
            });
        }

        // Disable Process Btn for history view
        // if (historyId) {
        //     processBtn.disabled = true;
        //     processBtn.textContent = "[ Chế Độ Xem Lịch Sử ]";
        // } else {
        //     processBtn.textContent = "Chạy Thuật Toán Nhận Diện";
        // }

        // Show Text Tab by default
        tabClick(resultTabText, contentText);
        downloadJsonBtn.disabled = false;

        // Share Link Setup
        if (historyId) {
            shareBtn.dataset.id = historyId;
            shareBtn.disabled = false;
        } else {
            shareBtn.disabled = true;
        }
    }

    // Tab logic
    function tabClick(activeTab, activeContent) {
        [resultTabText, resultTabJson, resultTabVisual].forEach(t => t.classList.remove('active'));
        [contentText, contentJson, contentVisual].forEach(c => c.classList.remove('active'));

        activeTab.classList.add('active');
        activeContent.classList.add('active');
    }

    resultTabText.addEventListener('click', () => tabClick(resultTabText, contentText));
    resultTabJson.addEventListener('click', () => tabClick(resultTabJson, contentJson));
    resultTabVisual.addEventListener('click', () => tabClick(resultTabVisual, contentVisual));

    // Download JSON
    downloadJsonBtn.addEventListener('click', () => {
        if (!resultData) return;
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(resultData, null, 2));
        const btn = document.createElement('a');
        btn.href = dataStr;
        btn.download = (currentFile ? currentFile.name : 'result') + '_graph.json';
        document.body.appendChild(btn);
        btn.click();
        document.body.removeChild(btn);
        log('Đã tải xuống file JSON.');
    });

    shareBtn.addEventListener('click', () => {
        const historyId = shareBtn.dataset.id;
        if (!historyId) return;
        const shareUrl = window.location.origin + window.location.pathname + '?view=' + historyId;
        navigator.clipboard.writeText(shareUrl).then(() => {
            alert("Đã sao chép Link chia sẻ kết quả vào Clipboard!");
        }).catch(err => {
            prompt("Link chia sẻ của bạn:", shareUrl);
        });
    });

    // History Redirect
    btnHistory.addEventListener('click', () => {
        window.location.href = '/history';
    });

    // Share Init
    async function loadShared() {
        const params = new URLSearchParams(window.location.search);
        const shareId = params.get('view');
        if (shareId) {
            log('Đang tải dữ liệu chia sẻ...');
            try {
                const res = await fetch('/api/v1/history/shared/' + shareId);
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail);

                const item = data.data;
                const jsonRes = await fetch('/' + item.result_json_path);
                const jsonData = await jsonRes.json();

                document.getElementById('upload-area').style.display = 'none';
                previewContainer.style.display = 'flex';
                previewImg.src = '/' + item.original_path;

                renderResult(jsonData, '/' + item.result_image_path, item.id);
                log(`Đã nạp báo cáo dữ liệu chia sẻ: ${item.filename}`, 'success');
            } catch (err) {
                log(`Lỗi tải báo cáo chia sẻ: ${err.message}`, 'error');
            }
        }
    }

    // Init
    checkAuth();
    loadShared();
    log('Hệ thống sẵn sàng.');
});
