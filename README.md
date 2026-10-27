# DBSCAN / OPTICS / HDBSCAN Demo

Demo Streamlit để trực quan hoá và so sánh ba thuật toán phân cụm ở 2D/3D.

## Yêu cầu

Python 3.10 trở lên, có sẵn `pip` và `venv`.

## Cài đặt

Mở PowerShell và chạy:

```powershell
cd D:\data-mining
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy demo

Tại thư mục `D:\data-mining`, chạy:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Mở [http://localhost:8501](http://localhost:8501) nếu trình duyệt chưa tự mở. Chọn dataset và tab thuật toán để bắt đầu.

Nhấn `Ctrl+C` trong terminal để dừng demo.
