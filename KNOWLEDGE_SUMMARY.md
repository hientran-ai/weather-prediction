# Tổng hợp kiến thức từ WeatherIQ

## 1. Bài toán được mô hình hóa như thế nào?

Mỗi dòng ở ngày `t` chứa thời tiết hiện tại và lịch sử gần nhất. Target là `tempmax(t+1)`. Vì vậy:

- Khi train, dòng cuối không có target và phải bị loại.
- Khi inference, dòng cuối lại chính là dòng quan trọng nhất và phải được giữ.
- Pipeline train và inference dùng chung cách tạo feature, nhưng khác nhau ở việc có tạo target hay không.

Đây là lý do `feature_engineering(..., include_target=True/False)` được tách rõ ràng.

## 2. Vì sao không được shuffle time series?

Random split có thể đưa dữ liệu tương lai vào tập train và làm metric đẹp giả tạo. Project chia tuần tự:

`70% quá khứ → train | 15% tiếp theo → validation | 15% mới nhất → test`

- Train dùng để học tham số.
- Validation dùng để chọn hyperparameter và chọn Ridge/LightGBM.
- Test chỉ dùng để báo cáo cuối cùng.

Không được chọn model bằng test set vì khi đó test đã tham gia quyết định và không còn là đánh giá độc lập.

## 3. Feature thời gian quan trọng

- Lag: giá trị của 1–3 ngày trước.
- Rolling: trung bình/độ lệch chuẩn/tổng mưa của một cửa sổ quá khứ.
- Difference: mức thay đổi so với ngày trước.
- Cyclical encoding: `sin/cos` giúp model hiểu tháng 12 gần tháng 1.
- Domain feature: mùa mưa, biên độ nhiệt, độ dài ban ngày, tương tác bức xạ–mây.

Rolling feature phải dùng `shift(1)` trước khi rolling để không vô tình đọc chính ngày đang dự báo.

## 4. Tránh data leakage

Các nguyên tắc đang áp dụng:

- Không shuffle dữ liệu.
- Không dùng test để chọn model.
- Không dùng nội suy hai chiều vì nó có thể đọc giá trị tương lai; chỉ forward-fill từ quá khứ.
- Scaler của Ridge chỉ `fit` trên train rồi mới `transform` validation/test.
- Dòng inference được tái tạo sau khi người dùng sửa input, tránh giữ feature dẫn xuất cũ.

## 5. Đọc metric đúng cách

- MAE: sai lệch tuyệt đối trung bình, dễ diễn giải theo °C.
- RMSE: phạt lỗi lớn mạnh hơn MAE, phù hợp khi muốn hạn chế dự báo lệch xa.
- R²: mức biến thiên được giải thích; có thể âm nếu model kém hơn dự báo hằng số.

MAE không phải khoảng tin cậy. Không nên hiển thị “dự báo ± MAE” như một xác suất chắc chắn. Muốn có khoảng dự báo nên dùng quantile regression, conformal prediction hoặc phân phối residual đã hiệu chỉnh.

## 6. Vì sao cần baseline?

Một model ML chỉ có giá trị nếu thắng được phương án đơn giản. Baseline hiện tại dùng `tempmax hôm nay` để đoán ngày mai. Trong lần chạy notebook cũ trên dữ liệu mẫu, Ridge và LightGBM thắng baseline, nhưng Ridge hơi tốt hơn LightGBM; độ phức tạp cao hơn không tự động đồng nghĩa chính xác hơn. Sau khi đổi chiến lược split, cần train lại để lấy bộ metric mới.

## 7. Session state và vòng đời model trong Streamlit

Streamlit chạy lại script sau mỗi tương tác. Do đó ứng dụng phải theo dõi:

- fingerprint của CSV hiện tại;
- fingerprint của dữ liệu đã dùng để train model;
- model/scaler/feature schema tương ứng.

Khi file đổi, model cũ phải bị xóa. Việc train được đặt sau nút bấm để một rerun thông thường không vô tình chạy lại tác vụ nặng.

## 8. Hướng nâng cấp tiếp theo

1. Walk-forward backtesting qua nhiều cửa sổ thời gian thay cho một holdout duy nhất.
2. Prediction interval bằng conformal prediction hoặc LightGBM quantile objective.
3. Theo dõi drift dữ liệu và sai số theo mùa.
4. Lưu model cùng metadata: schema, khoảng dữ liệu, phiên bản package và metric.
5. Tách training thành job nền nếu dữ liệu lớn.
6. Thêm kiểm thử end-to-end cho giao diện và CI trước khi deploy.
