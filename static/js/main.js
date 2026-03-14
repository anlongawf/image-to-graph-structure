document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
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
    const downloadJsonBtn = document.getElementById('download-json');
    const shareBtn = document.getElementById('share-btn');
    const noResultVisual = document.getElementById('no-result-visual');
    const resultImg = document.getElementById('result-img');
    const resultImgContainer = document.getElementById('result-img-container');

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
    async function checkAuth() {
        const token = localStorage.getItem('token');
        if (token) {
            try {
                const res = await fetch('/api/v1/auth/me', {
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                if (!res.ok) throw new Error('Token expired');
                const user = await res.json();
                
                if (user) {
                    btnLogin.classList.add('hidden');
                    btnLogout.classList.remove('hidden');
                    btnHistory.classList.remove('hidden');
                    userGreeting.textContent = `Chào, ${user.email.split('@')[0]}`;
                    userGreeting.classList.remove('hidden');
                } else {
                    handleLogout();
                }
            } catch (err) {
                handleLogout();
            }
        } else {
            handleLogout();
        }
    }

    function handleLogout() {
        localStorage.removeItem('token');
        btnLogin.classList.remove('hidden');
        btnLogout.classList.add('hidden');
        btnHistory.classList.add('hidden');
        userGreeting.classList.add('hidden');
    }

    // Auth Modal Logic
    btnLogin.addEventListener('click', () => { authModal.classList.remove('hidden'); });
    closeBtn.addEventListener('click', () => { authModal.classList.add('hidden'); });
    window.addEventListener('click', (e) => { if (e.target == authModal) authModal.classList.add('hidden'); });

    authToggle.addEventListener('click', (e) => {
        e.preventDefault();
        isRegisterMode = !isRegisterMode;
        authTitle.textContent = isRegisterMode ? 'Đăng Ký' : 'Đăng Nhập';
        authSubmitBtn.textContent = isRegisterMode ? 'Đăng Ký' : 'Đăng Nhập';
        authToggle.textContent = isRegisterMode ? 'Đã có tài khoản? Đăng nhập' : 'Chưa có tài khoản? Đăng ký ngay';
        authError.classList.add('hidden');
    });

    authSubmitBtn.addEventListener('click', async () => {
        const email = authEmail.value;
        const password = authPassword.value;
        if (!email || !password) {
            authError.textContent = "Vui lòng nhập đủ thông tin";
            authError.classList.remove('hidden');
            return;
        }

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
            authModal.classList.add('hidden');
            checkAuth();
            log(`Đã đăng nhập thành công.`, 'success');
        } catch (err) {
            authError.textContent = err.message;
            authError.classList.remove('hidden');
        }
    });

    btnLogout.addEventListener('click', () => {
        handleLogout();
        log('Đã đăng xuất.');
    });

    function log(message, type = "info", skipTimeFormat = false) {
        const div = document.createElement('div');
        let textContent = message;
        
        if (!skipTimeFormat) {
            const time = new Date().toLocaleTimeString();
            textContent = `[${time}] ${message}`;
        }

        div.textContent = textContent;
        if (type === 'error') div.classList.add('text-red-400');
        else if (type === 'success') div.classList.add('text-green-400');
        else if (skipTimeFormat) div.classList.add('text-gray-400');
        else div.classList.add('text-blue-400');

        statusLog.appendChild(div);
        statusLog.scrollTop = statusLog.scrollHeight;
    }

    // Drag and drop events
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-blue-500', 'bg-blue-50');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('border-blue-500', 'bg-blue-50');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-blue-500', 'bg-blue-50');
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
        previewContainer.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        processBtn.disabled = true;
        log('Đã xóa ảnh.');
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            log('Vui lòng chọn file hình ảnh hợp lệ (.png, .jpg)', 'error');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadArea.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            processBtn.disabled = false;
            log(`Đã tải ảnh: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
        };
        reader.readAsDataURL(file);
    }

    // Process Image
    processBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const token = localStorage.getItem('token');
        if (!token) {
            log('Bạn cần ĐĂNG NHẬP để chạy thuật toán.', 'error');
            authModal.classList.remove('hidden');
            return;
        }

        const formData = new FormData();
        formData.append('file', currentFile);

        log('Đang gửi ảnh lên máy chủ...');
        processBtn.disabled = true;
        spinner.classList.remove('hidden');

        try {
            const response = await fetch('/api/v1/extract/', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Lỗi không xác định');

            log(`Xử lý thành công trong ${data.execution_time_ms.toFixed(0)} ms`, 'success');
            renderResult(data.data, data.result_image_path, { id: data.id, is_public: false, is_owner: true });

        } catch (error) {
            log(`Lỗi: ${error.message}`, 'error');
        } finally {
            processBtn.disabled = false;
            spinner.classList.add('hidden');
        }
    });

    function renderResult(data, imgPath, historyInfo) {
        resultData = data || {};
        const nodes = Array.isArray(resultData.nodes) ? resultData.nodes : [];
        const edges = Array.isArray(resultData.edges) ? resultData.edges : [];

        let html = `<h3 class="text-lg font-bold mb-2">1. Tổng quan đỉnh (Nodes)</h3>`;
        html += `<p class="mb-4">Tìm thấy <b>${nodes.length}</b> đỉnh: ${nodes.map(n => `<span class="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-bold mr-1">${n.label}</span>`).join('')}</p>`;
        html += `<h3 class="text-lg font-bold mb-2">2. Mối liên kết (Edges)</h3>`;
        if (edges.length === 0) {
            html += `<p class="text-gray-500 italic">Không tìm thấy đường nối nào.</p>`;
        } else {
            html += `<ul class="space-y-2">`;
            edges.forEach(edge => {
                const weight = edge.weight ? `<span class="text-green-600 font-bold ml-2">(Trọng số: ${edge.weight})</span>` : '';
                html += `<li class="flex items-center"><span class="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center font-bold text-sm mr-2">${edge.from}</span> <svg class="w-4 h-4 mx-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg> <span class="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center font-bold text-sm ml-2">${edge.to}</span> ${weight}</li>`;
            });
            html += `</ul>`;
        }
        textOutput.innerHTML = html;

        // Always try to show the algorithm visual if available
        if (imgPath) {
            resultImg.src = imgPath;
            resultImg.classList.remove('hidden');
            noResultVisual.classList.add('hidden');
        } else {
            resultImg.classList.add('hidden');
            noResultVisual.classList.remove('hidden');
        }

        // Render JSON
        jsonOutput.textContent = JSON.stringify(data, (key, value) => key === 'logs' ? undefined : value, 2);

        // System Logs
        statusLog.innerHTML = '';
        if (Array.isArray(resultData.logs)) {
            resultData.logs.forEach(l => log(l, 'info', true));
        }

        downloadJsonBtn.disabled = false;
        
        // Share Button Logic
        if (historyInfo) {
            shareBtn.dataset.id = historyInfo.id;
            shareBtn.dataset.isPublic = historyInfo.is_public;
            shareBtn.dataset.isOwner = historyInfo.is_owner;
            shareBtn.disabled = false;
            updateShareBtnUI(historyInfo.is_public, historyInfo.is_owner);
        } else {
            shareBtn.disabled = true;
        }
    }

    function updateShareBtnUI(isPublic, isOwner) {
        if (!isOwner) {
            shareBtn.innerHTML = `Sao chép Link`;
            shareBtn.className = "flex-1 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors font-medium";
            return;
        }
        
        if (isPublic) {
            shareBtn.innerHTML = `Đang Chia sẻ (Hủy)`;
            shareBtn.className = "flex-1 px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors font-medium";
        } else {
            shareBtn.innerHTML = `Bấm để Chia sẻ`;
            shareBtn.className = "flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors font-medium";
        }
    }



    downloadJsonBtn.addEventListener('click', () => {
        if (!resultData) return;
        const blob = new Blob([JSON.stringify(resultData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `graph_${Date.now()}.json`;
        a.click();
    });

    shareBtn.addEventListener('click', async () => {
        const id = shareBtn.dataset.id;
        const isPublic = shareBtn.dataset.isPublic === 'true';
        const isOwner = shareBtn.dataset.isOwner === 'true';
        const token = localStorage.getItem('token');

        if (isOwner && token) {
            try {
                const res = await fetch(`/api/v1/history/${id}/toggle-share`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail);
                
                shareBtn.dataset.isPublic = data.is_public;
                updateShareBtnUI(data.is_public, true);
                
                if (data.is_public) {
                    const url = `${window.location.origin}${window.location.pathname}?view=${id}`;
                    navigator.clipboard.writeText(url).then(() => alert('Đã công khai và sao chép link chia sẻ!'));
                } else {
                    alert('Đã hủy chia sẻ. Chỉ bạn mới có thể xem kết quả này.');
                }
            } catch (err) {
                alert('Lỗi: ' + err.message);
            }
        } else {
            const url = `${window.location.origin}${window.location.pathname}?view=${id}`;
            navigator.clipboard.writeText(url).then(() => alert('Đã sao chép link chia sẻ!'));
        }
    });

    btnHistory.addEventListener('click', () => { window.location.href = '/history'; });

    // Load shared view
    async function loadShared() {
        const params = new URLSearchParams(window.location.search);
        const viewId = params.get('view');
        if (viewId) {
            log('Đang tải kết quả...');
            const token = localStorage.getItem('token');
            const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
            
            try {
                const res = await fetch(`/api/v1/history/shared/${viewId}`, { headers });
                const data = await res.json();
                
                if (!res.ok) {
                    if (res.status === 403) {
                        log('Bạn không có quyền xem kết quả này. Vui lòng đăng nhập hoặc yêu cầu chủ sở hữu chia sẻ.', 'error');
                        alert(data.detail);
                    } else if (res.status === 404) {
                        log('Không tìm thấy kết quả (ID có thể đã bị xóa hoặc không tồn tại).', 'error');
                        alert('Kết quả phân tích này không tồn tại hoặc đã bị xóa.');
                    } else {
                        throw new Error(data.detail || 'Lỗi không xác định');
                    }
                    return;
                }

                renderResult(data.data, '/' + data.history.result_image_path, {
                    id: data.history.id,
                    is_public: data.history.is_public,
                    is_owner: data.history.is_owner
                });
                previewImg.src = '/' + data.history.original_path;
                uploadArea.classList.add('hidden');
                previewContainer.classList.remove('hidden');
                log('Đã tải kết quả thành công.', 'success');
            } catch (err) {
                log(err.message, 'error');
            }
        }
    }

    checkAuth();
    loadShared();
    log('Hệ thống sẵn sàng.');
});
