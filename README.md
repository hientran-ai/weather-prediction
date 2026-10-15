# WeatherIQ — dự báo thời tiết Cà Mau

Ứng dụng Streamlit dùng dữ liệu thời tiết lịch sử để dự báo `tempmax` của ngày kế tiếp. Hai mô hình Ridge và LightGBM được so sánh với baseline “nhiệt độ cao nhất hôm nay”.

## Chạy project

Yêu cầu Python 3.11 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Sau khi mở ứng dụng:

1. Upload CSV thời tiết.
2. Kiểm tra thông báo xác thực dữ liệu.
3. Bấm **Train Model**.
4. Xem kết quả test và chuyển sang tab **Dự đoán**.

CSV phải có tối thiểu các cột `datetime`, `tempmax`, `tempmin`, `temp`, `precip`, `sealevelpressure`; nên có ít nhất 100 ngày liên tục. Các cột tùy chọn như `humidity`, `windspeed`, `cloudcover`, `uvindex`, `dew`, `sunrise`, `sunset` sẽ được dùng nếu có.

## Kiểm thử

```powershell
python -m pytest
```

Các test tập trung vào điều kiện quan trọng nhất của time series: target phải thuộc ngày kế tiếp, inference phải giữ ngày mới nhất, feature dẫn xuất phải được tính lại và ba tập dữ liệu không được chồng thời gian.

## Kiến trúc

- `app.py`: cấu hình trang và điều phối giao diện.
- `components/`: sidebar và bốn tab Streamlit.
- `utils/pipeline.py`: validation, preprocessing, feature engineering, split và inference thuần Python.
- `utils/train.py`: huấn luyện, tuning và đánh giá model.
- `utils/predict.py`: gọi model đã chọn để dự báo.
- `tests/`: regression tests cho pipeline dữ liệu.

`temp 3.ipynb` là notebook khám phá lịch sử và có thể dùng cách chia dữ liệu cũ; mã nguồn trong `utils/` mới là pipeline chuẩn của ứng dụng hiện tại.

Xem [KNOWLEDGE_SUMMARY.md](KNOWLEDGE_SUMMARY.md) để hiểu các quyết định ML và những hướng nâng cấp tiếp theo.
