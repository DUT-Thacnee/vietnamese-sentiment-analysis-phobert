# Phân tích Cảm xúc Bình luận Tiếng Việt bằng mô hình PhoBERT
Dự án này là một hệ thống phân tích cảm xúc (Tích cực/Tiêu cực) cho các bình luận sản phẩm bằng tiếng Việt, được xây dựng trong khuôn khổ môn học Trí tuệ Nhân tạo. Dự án sử dụng mô hình ngôn ngữ **PhoBERT** và thực hiện toàn bộ quy trình từ thu thập, tiền xử lý dữ liệu cho đến huấn luyện, đánh giá và triển khai một giao diện demo.

---

## 📋 Mục lục
- [Tổng quan](#-tổng-quan)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Quy trình thực hiện](#-quy-trình-thực-hiện)
  - [1. Thu thập dữ liệu](#1-thu-thập-dữ-liệu)
  - [2. Tiền xử lý dữ liệu](#2-tiền-xử-lý-dữ-liệu)
  - [3. Huấn luyện mô hình](#3-huấn-luyện-mô-hình)
- [Kết quả và Đánh giá](#-kết-quả-và-đánh-giá)
- [Giao diện Demo](#-giao-diện-demo)
- [Cách chạy dự án](#-cách-chạy-dự-án)
- [Hướng phát triển trong tương lai](#-hướng-phát-triển-trong-tương-lai)
- [Lời cảm ơn](#-lời-cảm-ơn)

---

## 🌟 Tổng quan

Trong bối cảnh bùng nổ của thương mại điện tử, việc hiểu được phản hồi của khách hàng là yếu tố sống còn đối với doanh nghiệp. Dự án này giải quyết bài toán phân loại các bình luận sản phẩm trên trang `Thegioididong.com` thành hai nhãn cảm xúc: **Tích cực (Positive)** và **Tiêu cực (Negative)**.

Mục tiêu chính là xây dựng một mô hình chính xác và đáng tin cậy, giúp tự động hóa quy trình phân tích phản hồi, từ đó cung cấp những insight giá trị cho doanh nghiệp.

**Workflow của hệ thống:**
`Thu thập bình luận` -> `Tiền xử lý văn bản` -> `Vector hóa (Embedding)` -> `Phân loại (Classifier)` -> `Kết quả (Positive/Negative)`

---

## 🤖 Kiến trúc mô hình

Mô hình được xây dựng dựa trên kiến trúc của **PhoBERT**, một mô hình Transformer được pre-train cho tiếng Việt bởi VinAI Research.
- **Base Model**: `vinai/phobert-base`
- **Kiến trúc tùy chỉnh**: Thêm một lớp `Dropout` (p=0.3) để chống overfitting và một lớp `Linear` để thực hiện phân loại nhị phân (2 classes).
- **Số tham số**: ~135 triệu.

 <!-- Hướng dẫn: Bạn có thể dùng ảnh ở trang 27 trong báo cáo -->

---

## 🛠️ Quy trình thực hiện

### 1. Thu thập dữ liệu
- **Nguồn**: Bình luận sản phẩm từ trang web [Thegioididong.com](https://www.thegioididong.com/).
- **Công cụ**: Sử dụng **Selenium** để tự động hóa trình duyệt, mô phỏng hành vi người dùng (cuộn trang, nhấn nút "Xem thêm") nhằm thu thập toàn bộ dữ liệu động (dynamic content).
- **Kết quả**: Thu thập được tập dữ liệu gồm các bình luận và số sao đánh giá tương ứng.

### 2. Tiền xử lý dữ liệu
Đây là bước quan trọng nhất để chuẩn hóa dữ liệu đầu vào. Quy trình gồm nhiều bước chi tiết:
- ✅ **Lowercasing**: Chuyển toàn bộ văn bản về chữ thường.
- ✅ **Removing Punctuation & Special Characters**: Loại bỏ dấu câu, emoji và các ký tự đặc biệt.
- ✅ **Removing Numbers**: Loại bỏ các chữ số không mang ý nghĩa cảm xúc.
- ✅ **Replacing Acronyms & Misspellings**: Chuẩn hóa các từ viết tắt, tiếng lóng (ví dụ: `sp` -> `sản phẩm`, `ko` -> `không`).
- ✅ **Spelling Correction**: Sử dụng một mô hình pre-trained khác để tự động sửa lỗi chính tả.
- ✅ **Word Segmentation**: Sử dụng `VnCoreNLP` để thực hiện tách từ tiếng Việt, một bước tối quan trọng cho các mô hình ngôn ngữ.
- ✅ **Removing Stopwords**: Loại bỏ các từ dừng (stopword) trong cả tiếng Việt và tiếng Anh.
- ✅ **Filtering**: Loại bỏ các bình luận quá ngắn hoặc quá dài, không mang đủ thông tin.

### 3. Huấn luyện mô hình
- **Phân chia dữ liệu**: Dataset được chia theo tỷ lệ **70% Training - 10% Validation - 20% Testing**.
- **Hàm mất mát (Loss Function)**: `Cross-Entropy Loss`.
- **Trình tối ưu (Optimizer)**: `AdamW`, được khuyến nghị cho các mô hình Transformer.
- **Siêu tham số (Hyperparameters)**:
  - **Learning Rate**: `2e-5`
  - **Batch Size**: `16`
  - **Epochs**: `5`
- **Model Selection**: Mô hình có kết quả tốt nhất trên tập validation được lưu lại để đánh giá cuối cùng. Mô hình tốt nhất đạt được sau **epoch thứ 2** với độ chính xác trên tập validation là **88.99%**.

---

## 📊 Kết quả và Đánh giá

Mô hình cuối cùng được đánh giá trên tập Test (dữ liệu chưa từng thấy).

**Overall Accuracy: 89%**

**Báo cáo phân loại (Classification Report):**

| Class    | Precision | Recall | F1-Score | Support |
| :------- | :-------- | :----- | :------- | :------ |
| 0 (Tiêu cực) | 0.85      | 0.89   | 0.87     | 1195    |
| 1 (Tích cực) | 0.92      | 0.89   | 0.90     | 1659    |
| **Accuracy** |           |        | **0.89** | **2854**|
| **Macro Avg**  | **0.89**  | **0.89** | **0.89** | **2854**|
| **Weighted Avg**| **0.89**  | **0.89** | **0.89** | **2854**|

**Nhận xét:**
- Mô hình đạt độ chính xác tổng thể **89%**, một kết quả rất tốt.
- Các chỉ số F1-score cân bằng (0.87 và 0.90) cho thấy mô hình hoạt động hiệu quả trên cả hai lớp cảm xúc.
- Mô hình cũng được kiểm tra với các trường hợp khó (edge cases) như câu phủ định phức tạp, câu chứa cảm xúc trái chiều và cho kết quả rất khả quan.

---

## 💻 Cách chạy dự án

### Yêu cầu
- Python 3.8+
- Pip

### Cài đặt
1.  Clone repository này về máy:
    ```bash
    git clone .....
    ```
2.  Di chuyển vào thư mục dự án:
    ```bash
    cd vietnamese-sentiment-analysis-phobert
    ```
3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
