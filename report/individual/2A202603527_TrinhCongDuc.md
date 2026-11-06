# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603527
- **Họ và tên:** Trịnh Công Đức
- **SEED** (in ở ô đầu notebook): 483 · Đề thử thách: #4 (Embedding — kNN với k = 1, 5, 25)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra: **15/15 ✅** (10 TODO + 5 giá trị tính tay khớp hash). Đủ 10 dự đoán, tất cả có dấu thời gian trước thí nghiệm tương ứng.

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (số của notebook, SEED 483; test = 775 tin, 96 spam; SUB = 40 tin cân bằng):

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.857 | 0.919 | 0.13 | 0 |
| 2. BoW + MLP (deep learning) | 0.951 | 0.974 | 0.34 | 0 |
| 3a. Encoder MiniLM + kNN (k = 5) | 0.903 | 0.952 | 62.75 | 0 |
| 3b. Encoder MiniLM + LogReg | 0.974 | 0.947 | 34.28 | 0 |
| 3c. Encoder MiniLM + MLP head | 0.974 | 0.950 | 47.66 | 0 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | — | 0.000* | 2618.47 | 0.0051 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | — | 0.000* | 1380.19 | 0.0510 |
| 4c. LLM nhỏ few-shot (`gpt-6-luna`) | — | 0.000* | 1473.01 | 0.0158 |

\* LLM chỉ chạy trên SUB. Cả 3 hàng LLM có `invalid = 40/40`: với `max_completion_tokens = 3`, model trả về chuỗi rỗng, `parse_label` trả `None` và tin bị tính là ham, nên TP = 0 và F1 = 0. Con số này cho thấy lỗi cấu hình, không đo năng lực phân loại của LLM (gọi thử cùng prompt với 16 token thì model trả đúng "spam").

- Trên đường Pareto: 1. BoW + LogReg và 2. BoW + MLP. Đạt SLA p95 < 50 ms: 1, 2, 3b, 3c.
- Với chỉ 40 tin, mỗi tin sai làm 1 − F1 đổi khoảng 2.5 điểm %, nên các hàng 2/3a/3b/3c trên SUB chỉ cách nhau 1 tin, coi như hòa.

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` lệch nhiều nhất:

1. **`llm_models` (15:59:52)**
   - **Đoán:** model lớn có F1 ngang hoặc cao hơn model nhỏ, chi phí gấp khoảng 10 lần, và LLM zero-shot khó thắng rõ 3b.
   - **Ra:** phần chi phí đúng ($0.00204 so với $0.00020 cho 40 tin, gấp 10 lần). Phần chất lượng thì không kiểm chứng được, vì cả hai model đều F1 = 0.000 với invalid = 40/40.
   - **Vì sao lệch:** em chỉ nghĩ đến model và prompt, không nghĩ đến cấu hình. 3 token output là quá ít cho `gpt-6-luna`, nên câu trả lời rỗng. Nếu dùng `parse_label_v2` (Bắt lỗi 3), lỗi này còn bị che đi hoàn toàn.
2. **`dl_curve` (14:13:21) + `dl_relu` (14:13:30)**
   - **Đoán:** MLP xấp xỉ logistic regression, không hơn rõ. Bỏ ReLU thì mạng tương đương LogReg.
   - **Ra:** MLP F1 test 0.951 so với 0.857 (recall 0.917 so với 0.750). Phần overfit thì đúng: val loss thấp nhất ở epoch 6. Bất ngờ là bản không ReLU cũng đạt 0.951.
   - **Vì sao lệch:** khoảng cách không đến từ tính phi tuyến mà từ cách huấn luyện. LogReg của em (GD full-batch, lr 0.5, 300 epoch) còn under-fit: ngay trên train, recall chỉ 0.832 và F1 0.906. Trong khi đó Adam + mini-batch train mạng tuyến tính 2 lớp tới gần hội tụ.
3. **`emb_tricky` (15:34:54)**
   - **Đoán:** 3b Encoder + LogReg đoán đúng nhiều hơn BoW ở 6 tin ngụy trang.
   - **Ra:** hòa, mỗi bên đúng 4/6.
     - Embedding đúng tin "brand new phone at no cost" (hiểu "no cost" ≈ "free") nhưng sai tin "chosen for a complimentary holiday", tin mà BoW lại đúng.
     - Cả hai cùng sai tin "account rewards expire tonight".
   - **Vì sao lệch:** em mặc định "hiểu nghĩa" thì luôn hơn "đếm từ". Thực tế, khi tin vẫn còn từ khóa mạnh (*reply*, *your* nằm trong top từ spam), BoW vẫn bắt được. Còn 6 tin là quá ít để kết luận bên nào hơn.

## 4. Phân tích 50 case khó

| Model | F1 SUB → F1 50 case khó | Acc 50 case khó |
| --- | --- | ---: |
| 1. BoW + LogReg | 0.919 → **0.333** | 52% |
| 2. BoW + MLP | 0.974 → **0.400** | 52% |
| 3a. Emb + kNN | 0.952 → 0.750 | 72% |
| 3b. Emb + LogReg | 0.947 → 0.640 | 64% |
| 3c. Emb + MLP head | 0.950 → 0.640 | 64% |
| 4a / 4d. LLM nhỏ (thường / phòng thủ) | 0.000 → 0.000 | 42% (đoán ham cả 50 tin vì rỗng) |

- **Tụt nhiều nhất:** hai model BoW, mất khoảng 0.58 F1.
  - Nhóm khó nhất với BoW + LogReg là spam giọng tự nhiên (đúng 10%), thao túng/lừa đảo (12%) và né từ khóa (17%). BoW + MLP còn 0% ở nhóm né từ khóa.
  - Ngược lại, BoW + LogReg đúng 100% ở "tưởng spam" và "nói về spam". Lý do: nhiều tin có từ kéo mạnh về ham (*i* w = −1.33, *me*, *my*, *that*), đủ bù lại từ khóa spam.
- **Embedding** chịu được né từ khóa và thao túng tốt hơn nhiều (kNN đúng 83% và 88%). Đổi lại, nó sai ở "nói về spam" (kNN chỉ đúng 25%), vì tin kể lại một tin trúng thưởng có *nghĩa* gần vùng spam.
- **Case cụ thể 1 — #31** "U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461": cả 7 model đều sai.
  - BoW: `tokenize` tách theo dấu chấm, nên *selected, cash, reward, call* biến thành các chữ cái đơn có trọng số gần 0. Từ khóa spam mạnh nhất *call* (w = +1.57) biến mất, chỉ còn các từ chức năng như *have, been, for, a*. Kết quả p(spam) = 0.24.
  - MiniLM: tokenizer cũng tách ra `s . e . l . e . c …`, nên vector không còn nghĩa "trúng thưởng".
- **Case cụ thể 2 — #14** "Hi, I'm working from home and making about £300 a day, happy to show you how…": cả 7 model đều sai.
  - BoW: tin toàn từ đời thường. *i* (w = −1.33) và *me* (−0.51), hai từ kéo về ham mạnh nhất, thắng *to*, *text* (+0.8). Còn *£300* không có trong vocab. Kết quả p(spam) = 0.03.
  - Embedding: câu có cấu trúc giống tin nhắn kể chuyện cá nhân, nên vector nằm cạnh các tin ham trò chuyện.
  - Đây là spam "giọng tự nhiên": không có tín hiệu bề mặt nào. Chỉ model hiểu *ý định* (rủ làm việc tại nhà kiếm tiền) mới bắt được.
- **Bonus — 5 case tự viết (`MY_CASES`, id 100–104)**, chạy với 3 model tất định (BoW + LogReg, kNN, 3b):
  - Số đúng: BoW 2/5, kNN 1/5, 3b 1/5. Model em chọn cho sản phẩm (3b) bị lừa 4/5.
  - Embedding bị lừa bởi tin "nói về spam" (#100 *mom keeps getting texts saying she won a free iPhone*): vector mang chủ đề "iPhone miễn phí", 4/5 hàng xóm là spam. BoW thoát nhờ *i* kéo mạnh về ham.
  - Cả ba đều thua #102 (giả bạn cùng lớp mời app kiếm tiền) và #104 (lừa xác nhận địa chỉ giao hàng). Lý do: tin ham trong train hay nói về app hay giao hàng, nên kiểu lừa mới rơi vào vùng ham. Embedding nhìn chủ đề chứ không nhìn ý định.
  - Bảng chi tiết và `why_tricky` từng case nằm ở ô ngay dưới `MY_CASES` trong notebook.

## 5. Thử thách cá nhân

**Đề #4 (SEED 483 % 6 = 3):** Embedding — chạy `knn_predict` với k = 1, 5, 25, báo F1, giải thích vì sao k quá nhỏ hoặc quá lớn đều có hại.

**Cách làm:**
- Ô ban đầu lỗi `NameError`: dùng `X_train`, `X_test`, `y_train`, `y_test`, `f1_score`, mà notebook không có các biến này và `f1_score` là của sklearn.
- Sửa lại: dùng `E_tr`/`E_te` (embedding đã encode ở Phần 3), `y_tr`/`y_te`, và hàm `metrics` tự viết.
- Chạy thêm 3 giá trị k trên 50 case khó để xem độ bền, rồi soi các tin bị đổi nhãn khi k tăng.

**Kết quả:**

| k | Precision test | Recall test | **F1 test** | FP / FN (test) | F1 · recall 50 case khó |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.886 | 0.969 | **0.925** | 12 / 3 | 0.717 · 0.655 |
| 5 | 0.845 | 0.969 | 0.903 | 17 / 3 | **0.750** · 0.724 |
| 25 | 0.823 | 0.969 | 0.890 | 20 / 3 | 0.737 · 0.724 |

**Giải thích:**
- **k quá nhỏ (k = 1) gây variance cao.** Nhãn chỉ dựa vào một hàng xóm, nên một tin train nhiễu là đủ đổi kết quả.
  - k = 1 thắng trên test vì 29% spam test có bản gần trùng trong train (cos ≥ 0.9), nên 1-NN chỉ chép nhãn của bản sao. Đây là kiểu trùng lặp gần mà notebook đã cảnh báo, làm điểm test lạc quan.
  - Trên 50 case khó (không có bản sao), k = 1 tệ nhất: recall 0.655 so với 0.724.
- **k quá lớn (k = 25) gây bias cao.** Vùng bỏ phiếu kéo cả tin ở xa vào và mọi phiếu nặng như nhau.
  - Spam là cụm dày hơn (cos tới hàng xóm gần nhất: trung vị 0.72, so với 0.53 của ham), nên khi vùng mở rộng, tin ham gần biên bị phiếu spam lấn át. FP tăng từ 12 lên 20, trong khi recall test không đổi.
  - Ví dụ: tin ham "Customer place i will call you." có hàng xóm ham gần như y hệt (cos 0.98) nhưng 15/25 hàng xóm là spam, nên bị chặn nhầm.
  - Cực đoan k → n thì mọi tin đều thành lớp đa số (ham 87.6%) và recall = 0.
- **Kết luận:** k = 5 là điểm cân bằng, tốt nhất trên case khó. Nên chọn k bằng tập validation (`E_va`), không theo test. Có thể bỏ phiếu có trọng số theo khoảng cách để giảm tác hại của k lớn.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (trung bình khoảng 116 tin/s), p95 < 50 ms/tin.

- **Đạt độ trễ:**
  - Đạt: 1. BoW + LogReg (0.13 ms), 2. BoW + MLP (0.34 ms), 3b (34 ms), 3c (48 ms, sát giới hạn).
  - Không đạt: 3a kNN (63 ms) và mọi LLM (1.4–2.6 giây/tin).
- **Đạt cả độ trễ lẫn chất lượng — em chọn 3b. Encoder MiniLM + LogReg:**
  - F1 test 0.974, cao nhất cùng 3c, và p95 34 ms (dư 16 ms so với SLA).
  - Trên 50 case khó đúng 64%, so với 52% của BoW.
- **Đánh đổi chấp nhận:**
  - Throughput cả lô chỉ khoảng 31 tin/s trên CPU Colab, nên cần khoảng 4 worker song song (nhiều hơn cho giờ cao điểm) hoặc 1 GPU.
  - Vẫn sai nhóm spam giọng tự nhiên (đúng 30%).
  - Nếu ưu tiên rẻ và nhanh tuyệt đối thì chọn 2. BoW + MLP: F1 0.951, hơn 30.000 tin/s, nhưng chỉ đúng 52% case khó.
- **Chi phí nếu dùng LLM cho mọi tin** (theo cột USD/1k tin):
  - Model nhỏ: 0.0051 × 10.000 = khoảng 51 USD/ngày, tức khoảng **1.530 USD/tháng**.
  - Model lớn: khoảng **15.300 USD/tháng**.
  - Kiểm tra chéo bằng tính tay #5 (100 token in + 1 token out, $0.10/$0.50 mỗi 1M token): $10.50 cho 1 triệu tin, tức khoảng 105 USD/ngày, khoảng 3.150 USD/tháng cho 10 triệu tin. Cột USD/1k của notebook ước lượng token bằng số từ / 0.75 nên thấp hơn.
  - Dù rẻ hay đắt, LLM vẫn **không đạt SLA** độ trễ.
- **Router:**
  - 3b xử lý mọi tin: p < 0.3 thì là ham, p > 0.7 thì là spam.
  - Chỉ tin lưỡng lự (p ∈ [0.3, 0.7]) mới gửi sang LLM, kèm prompt phòng thủ và kiểm tra đầu ra. Nếu LLM trả rỗng hoặc lỗi thì giữ nhãn của 3b.
  - Muốn p95 vẫn < 50 ms thì tỉ lệ tin chuyển sang LLM phải < 5%, hoặc xử lý LLM bất đồng bộ.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Em đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 5 — thử thách | Gỡ lỗi tên biến embedding, viết vòng lặp tính F1 cho các giá trị k | Code dùng `X_train`, `X_test`, `f1_score` | Chạy lỗi `NameError`: notebook không có các biến này, còn `f1_score` là của sklearn. Đã sửa (cùng Claude Code, dòng cuối) để dùng `E_tr`/`E_te` có sẵn và hàm `metrics` tự viết |
| Báo cáo | Gợi ý cấu trúc báo cáo và thiết kế router | Bản nháp báo cáo | Bản nháp có số liệu **không khớp** `lab1_report.json`. Đã bỏ và viết lại toàn bộ bằng số của notebook |
| Rà soát trước khi nộp (Claude Code) | Kiểm tra repo theo SUBMISSION/RUBRIC, tìm phần còn thiếu và làm nốt | Phát hiện: `HAND_CALC` để `None`, ô thử thách lỗi, câu trả lời Câu hỏi 4 nằm nhầm dưới Bắt lỗi 3, chưa có câu trả lời Bắt lỗi 3 và phần giải thích thử thách. AI soạn câu trả lời Bắt lỗi 3, giải thích thử thách, nhận xét `MY_CASES` và báo cáo này | 5 giá trị `HAND_CALC` lấy từ ảnh tay của em, khớp cả 5 hash. Ô thử thách và `MY_CASES` chạy lại trên máy local với cùng split SEED 483, sau khi kiểm chứng tái lập khớp Colab (BoW + LogReg F1 0.857, kNN k = 5 F1 0.903) |

**Điều em hiểu ra hôm nay:**

- Em đã nghĩ model phức tạp hơn sẽ thắng, nhưng các thí nghiệm có đối chứng cho thấy khác:
  - MLP hơn LogReg không phải nhờ ReLU, vì bản không ReLU cũng đạt 0.951.
  - Một LLM có thể ra F1 = 0 chỉ vì `max_tokens = 3`.
- Cấu hình và cách đánh giá quan trọng ngang model. Tập test cùng phân phối cho F1 0.97, nhưng 50 case khó kéo BoW xuống 0.33–0.40.
- Tập nhỏ (40 hay 6 tin) nhiễu đến mức một tin đã đổi thứ hạng, nên không được xếp hạng model theo vài điểm F1 lẻ.
