# Reflection — Lab 21

## 1. Điều làm tôi ngạc nhiên nhất

Điều làm tôi ngạc nhiên nhất là optimized prompt đưa target của base model từ 0.000 lên 0.765 mà không cần huấn luyện. Tôi cũng không kỳ vọng adapter đạt target 0.975 nhưng vẫn bị đánh FAILED vì regression giảm tới 0.1689. Kết quả này cho thấy nhìn riêng accuracy của tác vụ đích rất dễ dẫn đến quyết định deploy sai.

## 2. Phần mất nhiều thời gian nhất

Phần mất nhiều thời gian nhất là chạy ba cấu hình đối chứng trong NB4, vì mỗi cấu hình đều phải train đủ 30 bước để phép so sánh công bằng. Đây cũng là phần tôi dự đoán sẽ lâu, nhưng chi phí sinh văn bản và đánh giá nhiều lần cũng đáng kể hơn tôi nghĩ. Việc chạy trên Colab T4 giúp hoàn thành thí nghiệm mà không phụ thuộc GPU local.

## 3. Niềm tin đã thay đổi về fine-tuning

Trước lab, tôi cho rằng nếu fine-tune làm accuracy tác vụ tăng mạnh thì model đã tốt hơn và có thể deploy. Sau lab, tôi không còn tin kết luận đó nếu chưa có baseline prompt mạnh và regression gate. Adapter có thể rất giỏi một định dạng hẹp nhưng đồng thời đánh mất năng lực chung mà target metric không thể hiện.

## 4. Tôi dùng AI assistant như thế nào

Tôi dùng AI assistant để đọc cấu trúc repo, xác định thứ tự chạy Colab, kiểm tra ZIP artefact, đối chiếu số liệu giữa các tệp và hỗ trợ soạn báo cáo. Điểm cần cảnh giác là assistant ban đầu chỉ có thể hướng dẫn quy trình, không thể biết verdict trước khi có kết quả thật; các nhận định phải được kiểm tra lại bằng `results/*.json` và `runs.csv`. Tôi không dùng assistant để tự tạo số liệu hoặc thay đổi verdict.

## 5. Nếu fine-tune cho khách hàng thật

Bước đầu tiên của tôi sẽ là đóng băng một bộ đánh giá đại diện trước khi train, gồm tác vụ đích, regression, format và latency. Sau đó tôi sẽ đo cả naive prompt lẫn optimized prompt để biết fine-tuning có thực sự cần thiết không. Chỉ khi xác định được khoảng thiếu hụt còn lại, tôi mới thiết kế dữ liệu và cấu hình huấn luyện.
