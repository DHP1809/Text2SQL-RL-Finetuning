# Text2SQL Reinforcement Learning Finetuning Framework

Hệ thống chuyên dụng phục vụ quy trình huấn luyện hai giai đoạn (Two-Stage Training Flow) kết hợp giữa Tinh chỉnh giám sát (SFT) và Học tăng cường (Reinforcement Learning - GRPO) nhằm tối ưu hóa các mô hình ngôn ngữ lớn cho bài toán Text-to-SQL

---

## 🗂️ Cấu trúc và Mục đích các thư mục (Repository Structure)

### 1. `Finetuning/` (Mã nguồn Huấn luyện)
Tập trung toàn bộ mã nguồn, cấu hình và kịch bản huấn luyện phục vụ cho cả 2 giai đoạn tinh chỉnh mô hình:
* **LLamaFactory (SFT):** Triển khai kỹ thuật tinh chỉnh giám sát dựa trên chuỗi lý luận định hướng kế hoạch (`Plan-Guided Trajectories`), giúp mô hình nền tảng làm quen với việc sinh mã lập luận phức tạp trước khi tối ưu hóa chính sách.
* **EasyR1 (GRPO):** Thư viện chuyên trách triển khai thuật toán Học tăng cường **GRPO (Group Relative Policy Optimization)**, tối ưu hóa mô hình thông qua cơ chế chấm điểm thưởng tập trung (Reward System) mà không cần cấu hình mô hình Critic cồng kềnh, giúp tiết kiệm tối đa tài nguyên VRAM.

### 2. `arctic_text2sql_r1/` (Mã nguồn Đánh giá)
* Thư mục chuyên biệt chứa logic đánh giá hiệu năng (Evaluation Metrics) độc lập theo đúng phương pháp của framework **arctic_text2sql**.
* Chứa script `auto_evaluation.py` chịu trách nhiệm chạy thử nghiệm câu lệnh SQL đầu ra trực tiếp trên Database Sandbox để kiểm tra độ chính xác dựa trên kết quả thực thi (Execution-based Accuracy) thay vì chỉ so sánh chuỗi ký tự (Exact Match) một cách máy móc.

### 3. `data_process_pipeline/` (Đường ống Xử lý Dữ liệu)
Đường ống tiền xử lý dữ liệu đầu vào, tuân thủ nghiêm ngặt triết lý thiết kế và quy trình lọc dữ liệu để giúp mô hình hiểu sâu sắc cấu trúc cơ sở dữ liệu (Database Schema Understading):
**Data Filtering & Two-stage Execution Validation:** Thực thi kiểm thử hai bước trên mọi câu lệnh SQL trong tập dữ liệu gốc, tự động phát hiện và loại bỏ hoàn toàn các câu lệnh lỗi, rỗng, hoặc không tối ưu để đảm bảo tính sạch của tập dữ liệu huấn luyện.

---

## 🏋️ Quy trình Vận hành Tổng thể

1. **`data_process_pipeline`** làm sạch, lọc lỗi và đóng gói dữ liệu dựa trên phản hồi thực thi thực tế của Database Engine.
2. Dữ liệu chuẩn sau khi xử lý được nạp vào **`Finetuning/`** để chạy qua **LLamaFactory (SFT)** rồi tiếp tục căn chỉnh bằng **EasyR1 (GRPO)**.
3. Checkpoint mô hình sau huấn luyện sẽ được chuyển qua **`arctic_text2sql_r1/`** để đánh giá toàn diện các chỉ số và tỷ lệ thực thi thành công (Success Rate) trước khi đóng gói ra môi trường production.
