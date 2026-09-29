# DBSCAN Visualizer

Repository này là demo trực quan cho thuật toán phân cụm mật độ **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*). Mục tiêu là quan sát cách hai tham số `eps` và `min_samples` thay đổi số cụm, các điểm biên và các điểm nhiễu.

Demo gồm hai cách sử dụng:

- **Website Streamlit**: điều chỉnh tham số và xem kết quả phân cụm 2D/3D ngay trên trình duyệt.
- **Chương trình dòng lệnh**: tạo biểu đồ dữ liệu, kết quả DBSCAN và k-distance plot; có thể lưu thành ảnh PNG.

## Nội dung repository

| Tệp | Mô tả |
| --- | --- |
| `streamlit_app.py` | Website tương tác trực quan hoá DBSCAN ở 2D hoặc 3D. |
| `dbscan_demo.py` | Chương trình dòng lệnh tạo dữ liệu, chạy DBSCAN và vẽ biểu đồ. |
| `make_moons.csv` | Dữ liệu Make Moons gồm hai cụm cong lồng nhau. |
| `make-circles.ipynb` | Notebook minh hoạ/tạo cấu hình dữ liệu Make Circles. |
| `requirements.txt` | Danh sách thư viện Python cần cài. |

## Yêu cầu

- Python 3.10 trở lên.
- `pip` đi kèm Python.

## Cài đặt

Mở PowerShell tại thư mục repository rồi chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn việc kích hoạt môi trường ảo, chỉ cần chạy lệnh sau cho phiên PowerShell hiện tại rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Chạy website tương tác

```powershell
streamlit run streamlit_app.py
```

Streamlit sẽ mở trình duyệt tại [http://localhost:8501](http://localhost:8501).

Trên thanh bên, bạn có thể:

- Chọn dữ liệu mẫu hoặc tải tệp CSV.
- Chuyển giữa chế độ **2D** và **3D**; biểu đồ 3D có thể xoay và phóng to.
- Điều chỉnh `eps`, `min_samples` và random seed.
- Với CSV: chọn hai cột số làm X, Y; ở 3D có thể chọn cột Z hoặc để ứng dụng tạo Z từ X và Y.

## Chạy bằng dòng lệnh

```powershell
python dbscan_demo.py --dataset circles --eps 0.18 --min-samples 5
```

Lưu kết quả ra ảnh thay vì mở cửa sổ biểu đồ:

```powershell
python dbscan_demo.py --dataset blobs --eps 0.35 --min-samples 5 --save dbscan_blobs.png
```

Các bộ dữ liệu hỗ trợ là `moons`, `circles` và `blobs`.

Dataset `moons` đọc trực tiếp tệp `make_moons.csv` trong repository. Hai dataset `circles` và `blobs` được tạo trực tiếp bằng mã nguồn.

## Cách đọc kết quả

DBSCAN đánh dấu các điểm như sau:

- **Core point**: điểm lớn, viền trắng; có đủ ít nhất `min_samples` điểm trong bán kính `eps`.
- **Border point**: điểm nhỏ, mờ hơn; nằm gần một core point nhưng không đủ mật độ để tự trở thành core point.
- **Noise**: dấu `x` màu đen, nhãn `-1`; không thuộc về cụm nào.

Biểu đồ **k-distance** sắp xếp khoảng cách đến người láng giềng thứ `min_samples`. Có thể bắt đầu thử `eps` quanh đoạn đường cong đổi dốc mạnh ("elbow"). Dữ liệu được chuẩn hoá trước khi chạy DBSCAN, vì vậy `eps` là khoảng cách trong không gian đã chuẩn hoá.

## Tham số chính

- `eps`: bán kính lân cận. Giá trị quá nhỏ thường tạo nhiều noise; quá lớn có thể gộp các cụm khác nhau.
- `min_samples`: số điểm tối thiểu trong bán kính `eps` để một điểm được xem là core point (tính cả chính điểm đó).
- `--seed`: seed để tái tạo cùng dữ liệu ngẫu nhiên.
- `--save <tên-file.png>`: lưu biểu đồ của chương trình dòng lệnh vào tệp PNG.
