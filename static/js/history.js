document.addEventListener('DOMContentLoaded', async () => {
    const listContainer = document.querySelector('#history-list');
    const errorMsg = document.getElementById('error-msg');
    const userGreeting = document.getElementById('user-greeting');
    const btnLogout = document.getElementById('btn-logout');

    // Check Auth
    const token = localStorage.getItem('token');
    if (!token) {
        errorMsg.textContent = "Bạn cần Đăng nhập ở Trang chủ để xem lịch sử.";
        listContainer.innerHTML = '';
        userGreeting.style.display = 'none';
        btnLogout.style.display = 'none';
        return;
    }

    const payloadDecoded = JSON.parse(atob(token.split('.')[1]));
    userGreeting.textContent = `User ID: ${payloadDecoded.sub}`;

    btnLogout.addEventListener('click', () => {
        localStorage.removeItem('token');
        window.location.href = '/';
    });

    try {
        const res = await fetch('/api/v1/history/', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();

        listContainer.innerHTML = '';
        if (!res.ok) throw new Error(data.detail || "Không thể tải dữ liệu tự điển.");

        if (data.data.length === 0) {
            listContainer.innerHTML = '<li style="text-align:center; padding: 30px;">Bạn chưa thực hiện phân tích đồ thị nào.</li>';
        } else {
            data.data.forEach(item => {
                const li = document.createElement('li');
                li.className = 'history-item';
                li.innerHTML = `
                    <div class="history-imgs">
                        <div>
                            <div style="font-size: 0.7rem; color: #666; text-align: center; margin-bottom: 4px;">ẢNH GỐC</div>
                            <img src="/${item.original_path}" style="width: 70px; height: 70px; object-fit: cover; border-radius: 4px; border: 1px solid #ccc;" title="Ảnh Gốc">
                        </div>
                        <span style="font-size: 1.5rem; color: #888;">➡</span>
                        <div>
                            <div style="font-size: 0.7rem; color: #007bff; text-align: center; margin-bottom: 4px;">ẢNH KẾT QUẢ</div>
                            <img src="/${item.result_image_path}" style="width: 70px; height: 70px; object-fit: contain; border-radius: 4px; border: 1px solid #007bff; background: white;" title="Ảnh Kết Quả">
                        </div>
                    </div>
                    <div style="flex: 1; margin-left: 20px;">
                        <b style="color: #333; font-size: 1.1rem;">${item.filename}</b> <br> 
                        <small style="color: #666; display: inline-block; margin-top: 5px;">${new Date(item.created_at).toLocaleString()} &nbsp;|&nbsp; ${item.execution_time_ms}ms</small>
                    </div>
                    <a href="/?view=${item.id}" class="control-btn" style="text-decoration: none; padding: 10px 15px; display: inline-block; text-align: center;">Mở Lại 100%</a>
                `;
                listContainer.appendChild(li);
            });
        }
    } catch (err) {
        errorMsg.textContent = `Lỗi hệ thống: ${err.message}`;
        listContainer.innerHTML = '';
    }
});
