document.addEventListener('DOMContentLoaded', async () => {
    const listContainer = document.querySelector('#history-list');
    const errorMsg = document.getElementById('error-msg');
    const userGreeting = document.getElementById('user-greeting');
    const btnLogout = document.getElementById('btn-logout');

    const token = localStorage.getItem('token');
    if (!token) {
        errorMsg.textContent = "Bạn cần Đăng nhập ở Trang chủ để xem lịch sử.";
        errorMsg.classList.remove('hidden');
        listContainer.innerHTML = '<li class="p-12 text-center text-gray-500 italic">Vui lòng đăng nhập...</li>';
        return;
    }

    // Check Auth and get user info
    try {
        const res = await fetch('/api/v1/auth/me', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!res.ok) throw new Error('Token expired');
        const user = await res.json();
        userGreeting.textContent = `Chào, ${user.email.split('@')[0]}`;
    } catch (err) {
        localStorage.removeItem('token');
        window.location.href = '/';
        return;
    }

    btnLogout.addEventListener('click', () => {
        localStorage.removeItem('token');
        window.location.href = '/';
    });

    try {
        const res = await fetch('/api/v1/history/', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Không thể tải dữ liệu.");

        listContainer.innerHTML = '';
        if (data.data.length === 0) {
            listContainer.innerHTML = '<li class="p-12 text-center text-gray-500 italic">Bạn chưa thực hiện phân tích đồ thị nào.</li>';
        } else {
            data.data.forEach(item => {
                const li = document.createElement('li');
                li.className = 'p-6 hover:bg-gray-50 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-6';
                
                const date = new Date(item.created_at).toLocaleString('vi-VN');
                const statusColor = item.status === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700';
                const shareBadge = item.is_public ? '<span class="ml-2 px-2 py-0.5 rounded-full text-[10px] bg-blue-100 text-blue-700 font-bold uppercase border border-blue-200">Đã chia sẻ</span>' : '';
                
                li.innerHTML = `
                    <div class="flex items-center space-x-6">
                        <div class="flex-shrink-0 relative">
                            <img src="/${item.result_image_path}" class="w-20 h-20 object-contain rounded-lg border border-gray-200 bg-gray-50 shadow-sm" alt="Preview">
                            <span class="absolute -top-2 -right-2 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${statusColor} border border-white">
                                ${item.status}
                            </span>
                        </div>
                        <div>
                            <div class="flex items-center">
                                <h3 class="text-lg font-bold text-gray-900 leading-tight">${item.filename.split('_').slice(1).join('_')}</h3>
                                ${shareBadge}
                            </div>
                            <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-gray-500 mt-1">
                                <span class="flex items-center">
                                    <svg class="w-4 h-4 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                                    ${date}
                                </span>
                                <span class="flex items-center">
                                    <svg class="w-4 h-4 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                    ${item.execution_time_ms ? item.execution_time_ms.toFixed(0) : 0}ms
                                </span>
                            </div>
                        </div>
                    </div>
                    <div class="flex items-center space-x-3">
                        <a href="/?view=${item.id}" class="inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 transition-colors shadow-sm">
                            Xem chi tiết
                            <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 5l7 7-7 7"></path></svg>
                        </a>
                    </div>
                `;
                listContainer.appendChild(li);
            });
        }
    } catch (err) {
        errorMsg.textContent = `Lỗi hệ thống: ${err.message}`;
        errorMsg.classList.remove('hidden');
        listContainer.innerHTML = '';
    }
});
