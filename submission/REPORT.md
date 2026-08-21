# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đăng Tuyên  
**MSSV**: 2A202601622  
**Ngày**: 21/08/2026  
**Tier**: `T4`  
**Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: NVIDIA Tesla T4 16 GB

> Mọi số liệu trong báo cáo được lấy từ các tệp trong `results/`. Thí nghiệm dùng toàn bộ tập đánh giá, không bật `EVAL_LIMIT`.

---

## 1. Setup

| Hạng mục | Giá trị |
|---|---|
| Dataset | 250 ticket chăm sóc khách hàng tiếng Việt → JSON triage |
| Train / validation | 225 / 25, chia với seed 42 |
| `max_length` | 1024 theo preset T4; p95 đo được là 98 token và giá trị làm tròn được gợi ý là 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max steps | 2 / 30 |

`max_length=1024` lớn hơn mức 256 được gợi ý từ p95. Tôi giữ preset của tier T4 để thí nghiệm chính và ba đối chứng dùng cùng một cấu hình độ dài, đồng thời báo cáo rõ độ lệch thay vì xem 1024 là một con số suy đoán từ dữ liệu. Với corpus hiện tại có tối đa 101 token, 256 sẽ là lựa chọn tiết kiệm hơn nếu tối ưu lại pipeline.

**Template có giữ khối `<think>` không?** Có. `template_check.json` xác nhận cả thẻ mở và nội dung reasoning đều xuất hiện, với verdict `reasoning preserved — safe to train on traces`. Tuy nhiên, 250 câu trả lời huấn luyện của corpus mặc định chỉ chứa JSON, không có reasoning trace thật; vì vậy việc template hỗ trợ `<think>` không làm xuất hiện trace hợp lệ trong đầu ra.

---

## 2. Mask proof (NB1)

| Hạng mục | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |

Đoạn đầu của phần được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh",
 "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Kết quả này cho thấy loss chỉ giám sát phần assistant trả lời. Phần system và user nằm trong `masked_preview`, còn câu trả lời JSON nằm trong `supervised_preview`; do đó model không bị huấn luyện để học thuộc hoặc viết lại câu hỏi.

---

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3184.2 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1027.5 |
| (c) LoRA fine-tune | 0.9750 | 0.5889 | 1.0000 | 1446.0 |

Baseline (b) thực sự mạnh hơn (a): target tăng từ 0.000 lên 0.765 và format tăng từ 0.000 lên 1.000, trong khi regression không đổi. Ngoài ra, (b) còn nhanh hơn khoảng 2156.7 ms/mẫu. Tôi không sửa `OPTIMIZED_PROMPT`; SHA được gatekeeper xác nhận là `719e74d3b6232053`. Vì vậy phép so sánh không làm yếu prompt nhằm khiến fine-tune trông tốt hơn.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí | r | Trainable params | LR | Train loss | Target | Thời gian (s) | VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6259 | 0.975 | 931.6 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5382 | 0.970 | 826.0 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 960.5 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1030.8 | 7.09 |

### 4.1 — Vị trí adapter so với rank

`attn_only` có 32,456,704 tham số huấn luyện, chỉ ít hơn `correct` 8,192 tham số, tương đương khoảng 0.025%; đây là một đối chứng khớp ngân sách tốt. Trên target, `correct` đạt 0.975 và thắng nhẹ `attn_only` ở 0.970, dù `attn_only` có train loss thấp hơn, 0.5382 so với 0.6259. Như vậy thứ tự theo loss và thứ tự theo chất lượng target bị đảo ngược. Việc tăng rank của attention-only lên 283 không bù hoàn toàn cho việc chỉ đặt adapter ở q,v; bằng chứng này cho thấy vị trí phủ các lớp text-linear là đòn bẩy đáng tin cậy hơn việc chỉ tăng rank trong attention.

### 4.2 — Learning rate sai

`wrong_lr` chỉ giảm learning rate từ 1e-4 xuống 1e-5, còn vị trí, rank, số tham số và 30 bước đều giữ nguyên. Loss cuối tăng từ 0.6259 lên 1.5704 và target sụp từ 0.975 xuống 0.000; format cũng về 0.000. Nếu chỉ nhìn đường loss mà không biết LR, tôi có thể kết luận nhầm rằng dữ liệu, mask hoặc kiến trúc LoRA không học được. Đối chứng cho thấy nguyên nhân trực tiếp là dùng thang learning rate của full fine-tuning cho LoRA, khiến 30 bước không đủ tạo ra cập nhật hữu ích.

### 4.3 — QLoRA

QLoRA giảm peak VRAM từ 12.01 GB xuống 7.09 GB, tiết kiệm 4.92 GB, khoảng 41%. Đổi lại, target giảm từ 0.975 xuống 0.940, loss tăng từ 0.6259 lên 0.7058 và thời gian train tăng từ 931.6 lên 1030.8 giây. Trên T4 16 GB, bản LoRA fp16 đã vừa bộ nhớ nên phần tiết kiệm này không cần thiết cho thí nghiệm hiện tại. Số đo của tôi vì thế ủng hộ khuyến nghị không dùng QLoRA cho cấu hình Qwen3.5 này khi đủ VRAM, dù QLoRA vẫn là lựa chọn thực dụng nếu giới hạn phần cứng quan trọng hơn 0.035 điểm target và thời gian chạy.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
**Target Δ**: `+0.2100`  
**Regression Δ**: `-0.1689`  
**Valid trace rate**: `0.00`

Fine-tune cải thiện đúng tác vụ triage rất rõ: target tăng từ 0.765 lên 0.975 và vẫn giữ format ở 1.000. Tuy nhiên, regression giảm từ 0.7578 xuống 0.5889, tức giảm khoảng 0.169, lớn hơn nhiều so với ngưỡng cho phép 0.020. Latency cũng tăng từ 1027.5 lên 1446.0 ms/mẫu so với optimized prompt. Vì cổng đánh giá yêu cầu đồng thời cải thiện tác vụ đích mà không phá hỏng năng lực chung, kết quả phải là FAILED dù target rất cao. Nguyên nhân hợp lý nhất là huấn luyện chuyên biệt trên 225 ticket JSON mà không có dữ liệu replay kiến thức phổ thông, khiến model bị kéo quá mạnh về phân phối triage. `valid_trace_rate=0` cũng phù hợp với việc corpus huấn luyện không chứa reasoning trace thật. Tôi không nới ngưỡng để đổi verdict; bước tiếp theo hợp lý là thêm 1–5% replay data, huấn luyện lại và đo trên chính tập eval đã đóng băng.

---

## 6. Định tính — có cả ca fine-tune sai

NB5 lưu nhãn và dự đoán fine-tune theo từng mẫu nhưng NB2 chỉ lưu điểm tổng hợp, không lưu chuỗi dự đoán (b) theo từng mẫu. Vì vậy tôi ghi rõ giới hạn này thay vì tự dựng output của baseline (b).

| # | Ticket rút gọn | Nhãn đúng | (b) optimized prompt | (c) fine-tune | Nhận xét |
|---:|---|---|---|---|---|
| 1 | Trả lại chuột không dây, gấp, shop hỗ trợ tốt | `doi_tra / cao / chuột không dây / tich_cuc` | Không lưu output từng mẫu | Khớp đủ 4 trường, score 1.00 | ✅ FT đúng hoàn toàn |
| 2 | Hoàn tiền ốp lưng điện thoại, sớm, bực mình | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | Không lưu output từng mẫu | Khớp đủ 4 trường, score 1.00 | ✅ FT đúng hoàn toàn |
| 3 | Chưa thấy tiền bình giữ nhiệt, khi nào tiện, cảm ơn | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | Không lưu output từng mẫu | Dự đoán urgency `trung_binh`, score 0.75 | ❌ FT sai urgency |
| 4 | Áo khoác gió bị lỗi, khi nào tiện, cảm ơn | `san_pham_loi / thap / áo khoác gió / tich_cuc` | Không lưu output từng mẫu | Dự đoán urgency `trung_binh`, score 0.75 | ❌ FT sai urgency |
| 5 | Hoàn tiền nồi chiên, khi nào tiện, quá tệ | `hoan_tien / thap / nồi chiên không dầu / tieu_cuc` | Không lưu output từng mẫu | Dự đoán urgency `trung_binh`, score 0.75 | ❌ FT sai urgency |

Mẫu chung của ba ca tệ nhất là model đều dự đoán `trung_binh` thay vì `thap` cho cụm “khi nào tiện”. Intent và product vẫn đúng, nên lỗi tập trung ở ranh giới nhãn urgency chứ không phải model không hiểu loại yêu cầu. Điều này gợi ý cần kiểm tra cân bằng nhãn và bổ sung các biến thể diễn đạt urgency thấp trong dữ liệu train.

---

## 7. Kết luận và điều tôi học được

**Kết luận.** Tôi chưa nên deploy adapter này dù accuracy target đạt 0.975. So với base model dùng optimized prompt, adapter tăng 0.210 điểm target và tạo JSON hợp lệ ở toàn bộ 50 mẫu, nhưng làm regression giảm 0.1689 và tăng latency khoảng 418.5 ms/mẫu. Đây không phải một trao đổi chấp nhận được nếu hệ thống còn phải xử lý câu hỏi phổ thông ngoài triage. Thí nghiệm đối chứng cũng cho thấy không thể chọn cấu hình chỉ dựa vào train loss: `attn_only` có loss thấp hơn nhưng target vẫn thấp hơn `correct`. Đòn bẩy quan trọng nhất trong các run đã đo là learning rate và loss mask đúng; LR thấp hơn mười lần làm target sụp hoàn toàn, còn mask đúng bảo đảm model học câu trả lời thay vì prompt. Vị trí adapter cũng quan trọng hơn việc tăng rank một cách đơn thuần, vì rank 283 ở q,v không vượt được rank 16 phủ text-linear dù ngân sách tham số gần như bằng nhau. Nếu tiếp tục, tôi sẽ giữ nguyên tập đánh giá, thêm 1–5% replay data kiến thức phổ thông, tăng ví dụ cho urgency thấp và chỉ deploy khi cổng regression chuyển sang PASS.

**Ba điều tôi học được:**

1. Prompt tối ưu là một baseline bắt buộc: riêng việc đổi từ naive sang optimized prompt đã đưa target từ 0.000 lên 0.765 mà chưa fine-tune.
2. Train loss không đủ để xếp hạng adapter; `attn_only` có loss tốt hơn nhưng target kém hơn `correct`.
3. Target accuracy cao không đồng nghĩa sẵn sàng deploy; regression gate đã phát hiện mức suy giảm năng lực chung 0.1689.

**Nếu có thêm 2 giờ, tôi sẽ thử:** trộn 1–5% replay data kiến thức phổ thông vào tập train, giữ nguyên mask và cấu hình `correct`, sau đó chạy lại cùng 30 bước và đánh giá trên đúng checksum hiện tại. Tôi cũng sẽ lưu prediction của baseline (b) theo từng mẫu để phân tích trực tiếp các ca fine-tune thắng và thua.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [x] B5 Hugging Face Hub — https://huggingface.co/Tuyen062004/lab21-qwen35-4b-lora
