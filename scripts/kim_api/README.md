# Kiểm tra danh sách ảnh và quota trước khi chạy (không gọi API)

```
uv run python scripts/kim_api/scheduler.py --input-dir <thư_mục_ảnh_ch> --output-dir <thư_mục_xuất> --dry-run
```

# Chạy bình thường (tự ngủ khi chạm quota cho đến khi xong toàn bộ ảnh)

Cần chạy lại script khi khởi động máy, nhập TOKEN nếu không có `.env`.

```
uv run python scripts/kim_api/scheduler.py --input-dir <thư_mục_ảnh_ch> --output-dir <thư_mục_xuất> [--token <TOKEN>]
```

# Hoặc đẩy vào background bằng nohup

Cần chạy lại script khi khởi động máy, nhập TOKEN nếu không có `.env`.

```
nohup uv run python scripts/kim_api/scheduler.py --input-dir <thư_mục_ảnh_ch> --output-dir <thư_mục_xuất> [--token <TOKEN>] > scheduler.log 2>&1 &
```

# Xử lý lại request bị lỗi

```
nohup uv run python scripts/kim_api/scheduler.py --input-dir <thư_mục_ảnh> --output-dir <thư_mục_output> --retry-failed > scheduler.log 2>&1 &
```

1. Script sẽ quét toàn bộ các task đang bị FAILED trong database và tự động reset:

- Nếu ảnh đã upload thành công trước khi rớt mạng, chuyển về `UPLOADED` (không tốn thêm 1 request upload lại).
- Nếu ảnh chưa upload được chuyển về `PENDING`.
- Đặt lại `retry_count = 0` và xóa `last_error`.

2. Script tiếp tục chạy bình thường mà không làm ảnh hưởng đến các ảnh đã COMPLETED trước đó.

# Xem tiến độ hiện tại trong DB

```
uv run python scripts/kim_api/scheduler.py --output-dir <thư_mục_xuất> --status
```

# Dừng script

```
pkill -f "scheduler.py"
```
