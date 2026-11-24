# Kiểm tra danh sách ảnh và quota trước khi chạy (không gọi API)

```
uv run python scripts/kim_api/scheduler.py --input-dir <thư_mục_ảnh_ch> --output-dir <thư_mục_xuất> --dry-run
```

# Chạy bình thường (tự ngủ khi chạm quota cho đến khi xong toàn bộ ảnh)

```
uv run python scripts/kim_api/scheduler.py --input-dir <thư_mục_ảnh_ch> --output-dir <thư_mục_xuất>
```

# Hoặc đẩy vào background bằng nohup

```
nohup uv run python scripts/kim_api/scheduler.py --input-dir <thư_mục_ảnh_ch> --output-dir <thư_mục_xuất> > scheduler.log 2>&1 &
```

# Xem tiến độ hiện tại trong DB

```
uv run python scripts/kim_api/scheduler.py --output-dir <thư_mục_xuất> --status
```
