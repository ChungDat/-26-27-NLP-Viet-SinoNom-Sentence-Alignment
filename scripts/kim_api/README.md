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

# Xem tiến độ hiện tại trong DB

```
uv run python scripts/kim_api/scheduler.py --output-dir <thư_mục_xuất> --status
```

# Dừng script

```
pkill -f "scheduler.py"
```
