# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604091
- **Họ và tên:** Trương Văn Phong
- **SEED** (in ở ô đầu notebook): STUDENT_ID=2A202604091  →  SEED=8913

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ (ô báo cáo cuối đã chạy và in "Đã lưu lab1_report.json") |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

**Trạng thái bộ kiểm tra:** 14/15 đạt. Các hàm tự code ở Phần 1–3 (`vectorize`, `sigmoid`/`predict_proba`, `train_logreg`, `metrics`, `build_mlp`, `train_mlp`, `cosine_similarity`/`top_k_similar`, `knn_predict`) và 5/5 phép tính tay đều ✅. Đủ 10/10 dự đoán (`ml_f1`, `ml_lr`, `dl_curve`, `dl_relu`, `emb_search`, `emb_tricky`, `llm_models`, `llm_fewshot`, `tradeoff`, `hard_eval`).

**Phần chưa hoàn thành:**

- Check **4.1 + 4.4 (`call_openai` / `llm_classify_batch`) ❌** với lỗi `NameError: name 'OpenAI' is not defined`. Nguyên nhân: trong `call_openai` tôi gọi `OpenAI()` nhưng chưa `from openai import OpenAI` (gợi ý của template đặt import này ngay trong hàm). Cách sửa: thêm `from openai import OpenAI` vào đầu thân hàm `call_openai`.
- Tôi **không có API key** nên `HAS_LLM = False`: toàn bộ ô gọi LLM (Thí nghiệm 4.A, 4.B, hàng 4a/4b/4c/4d) bị bỏ qua. Vì vậy báo cáo này **không có số đo thực tế nào của LLM** (F1, độ trễ, chi phí đo được). Mọi con số về LLM bên dưới là **ước tính từ phép tính tay**, không phải kết quả chạy.

**Tính tay (ảnh `hand_calc.jpg`):**

| # | Đại lượng | Các bước | Kết quả |
| --- | --- | --- | --- |
| 1 | `baseline_accuracy` | Sau khi bỏ trùng còn 5169 tin, trong đó ham = 4516 → luôn đoán ham đúng 4516/5169 | **0.87** |
| 2 | `sigmoid_0` | σ(0) = 1 / (1 + e⁰) = 1/2 | **0.50** |
| 3 | `f1_example` (TP=80, FP=20, FN=40) | Precision = 80/(80+20) = 0.8; Recall = 80/(80+40) ≈ 0.667; F1 = 2·0.8·0.667/(0.8+0.667) ≈ 0.727 | **0.73** |
| 4 | `cosine_example` ([1,2,0] và [2,1,2]) | Tích vô hướng = 1·2 + 2·1 + 0·2 = 4; ‖a‖ = √(1+4+0) = √5 ≈ 2.236; ‖b‖ = √(4+1+4) = √9 = 3; cos = 4/(2.236·3) ≈ 0.596 | **0.60** |
| 5 | `llm_cost_per_day` (1.000.000 tin/ngày, 100 token vào + 1 token ra, giá $0.10 / $0.50 mỗi 1M) | Input: 1.000.000 × 100 = 100M token → 100 × $0.10 = $10; Output: 1.000.000 × 1 = 1M token → $0.50; tổng | **$10.50 / ngày** |

*Ghi chú:* trong ảnh, ở mục 1 tôi chỉ ghi kết quả 0.87; mục 3 mới viết đến bước precision = 0.8 (recall và F1 tính ở trên); ở mục 4 dòng ‖b‖ tôi viết nhầm "= 9" trong khi giá trị đúng là √9 = 3 (phép chia cuối cùng vẫn dùng 3 nên kết quả 0.596 → 0.6 là đúng).

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 của tôi (SEED 8913; test = 775 tin, SUB = 40 tin cân bằng 20 spam / 20 ham; độ trễ đo trên Colab, gọi từng tin một, từ văn bản thô → nhãn):

| Cách tiếp cận | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 1.000 | 0.832 | 0.908 | 0.919 | 10.1 | 6558 | 0.054 | 0.200 | 0 |
| 2. BoW + MLP (deep learning) | 0.977 | 0.905 | 0.940 | 0.947 | 23.2 | 12843 | 0.311 | 0.379 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.909 | 0.947 | 0.928 | 0.974 | 137.1 | 37.5 | 80.29 | 132.51 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.989 | 0.958 | **0.973** | 0.974 | 138.6 | 43.8 | 31.82 | **40.63** | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.968 | 0.968 | 0.968 | 0.974 | 138.1 | 44.5 | 37.73 | 62.50 | 0 |
| 4a/4b/4c. LLM (zero-shot / few-shot) | — | — | — | — | — | — | — | — | — *(không chạy: thiếu API key)* |

Nhận xét ngắn:

- **F1 test cao nhất là 3b (0.973)**, và 3b cũng là cách chất lượng cao nhất vẫn đạt p95 < 50 ms (40.63 ms). Chênh giữa 3a/3b/3c trên SUB bằng 0 (cùng 0.974), nên trên 40 tin không xếp hạng được; phải nhìn F1 trên 775 tin test.
- BoW + LogReg có **precision = 1.000 nhưng recall chỉ 0.832**: nó không chặn nhầm tin nào nhưng bỏ sót ~17% spam (ví dụ "Camera - You are awarded a SiPix Digital Camera! call 09061221066…" bị đoán là ham). Các từ đẩy về spam mạnh nhất: `call, txt, free, your, to, text, reply, stop, now, uk, www, or`; các từ kéo về ham: `i, me, my, it, lt, gt, that, how, but, in, ok, not`.
- MLP trên BoW (F1 0.940) hơn LogReg (0.908) khoảng 0.03 nhưng không "thắng rõ"; val loss thấp nhất ở epoch 5 trong 30 epoch (dấu hiệu overfit sớm). Bỏ ReLU chỉ giảm F1 xuống 0.934, đúng với dự đoán "gần như một mô hình tuyến tính".
- Thêm hidden layer trên embedding (3c vs 3b) **không giúp**: F1 0.968 so với 0.973, mà p95 còn xấu hơn (62.5 so với 40.6 ms).
- Pareto (trên F1 SUB): **1, 2, 3b**. Vượt SLA 50 ms: **3a (132.5 ms)** và **3c (62.5 ms)**.

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất (cả ba đều ghi bằng `predict(...)` trước khi chạy):

**(1) `hard_eval` — 50 case khó.**

- *Tôi đoán:* accuracy BoW còn ~50–60%, Embedding ~75–85%, LLM ~85–90%; nhóm khó nhất với BoW là "né từ khoá" và "tưởng spam nhưng không phải".
- *Thực tế:* BoW **50%**, BoW+MLP 56%, 3a **74%**, 3b **64%**, 3c **62%** (LLM không chạy). BoW đúng khoảng đoán (chạm cận dưới), nhưng cả ba model embedding đều **thấp hơn khoảng 75–85% tôi đoán**. Về nhóm: BoW đạt **100%** ở "tưởng spam nhưng không phải", ngược hoàn toàn với dự đoán; nhóm khó nhất với BoW là "spam giọng tự nhiên" (0%), "thao túng/lừa đảo" (12%) và "né từ khoá" (17%).
- *Vì sao lệch:* tôi nghĩ BoW sẽ bị từ khoá "WON/FREE/URGENT" trong tin bạn bè đánh lừa, nhưng thực tế BoW **thiên về đoán ham** (train chỉ có 463/3619 ≈ 12.8% là spam, và các từ `i, me, my` kéo mạnh về ham). Tính ngược từ F1 = 0.286 và accuracy = 0.50 trên 50 case (29 spam, 21 ham), BoW chỉ bắt được **5/29 spam** và chỉ nhầm 1 tin ham. Nên 100% ở các nhóm ham chỉ là hệ quả của việc "hầu như luôn đoán ham", không phải hiểu ngữ cảnh. Với embedding, tôi đã đánh giá quá cao khả năng "hiểu nghĩa": vector bị kéo theo chủ đề (tiền thưởng, trúng thưởng), nên các nhóm ham chứa từ khoá bị tụt (3a chỉ đúng 25% ở nhóm "nói về spam").

**(2) `emb_tricky` — 6 tin ngụy trang (BoW hay Encoder+LogReg đúng nhiều hơn?).**

- *Tôi đoán:* chỉ ghi "ham" — câu trả lời này không nêu được mô hình nào thắng nên thực chất là dự đoán không dùng được.
- *Thực tế:* BoW đúng **4/6**, Encoder+LogReg đúng **5/6**. Encoder+LogReg đoán đúng hai tin né từ khoá ("get a brand new phone at no cost…", "Last chance! Your account rewards expire tonight…") mà BoW bỏ sót; ngược lại, Encoder+LogReg **sai** tin "Your mobile number has been chosen for a complimentary holiday. Reply YES to confirm" mà BoW lại đúng.
- *Vì sao lệch:* BoW đúng tin đó nhờ các từ `your`, `reply` nằm trong nhóm đẩy về spam. Điều bất ngờ hơn: ở thí nghiệm 3.A, tìm kiếm theo nghĩa cho một truy vấn gần như cùng ý ("complimentary holiday, reply now to claim") trả về cả 6 lân cận đều là spam, nhưng đầu LogReg trên cùng embedding vẫn đoán ham cho tin này. Tức là "gần nghĩa với spam" chưa chắc nằm đúng phía của một mặt phẳng tuyến tính học từ 463 tin spam; bộ kNN thì vẫn dùng đúng cấu trúc lân cận đó.

**(3) `tradeoff` — đường Pareto và vạch SLA 50 ms.**

- *Tôi đoán:* Pareto gồm 3b và 1; kNN (3a) và LLM nằm bên phải vạch 50 ms.
- *Thực tế:* Pareto gồm **1, 2 và 3b**; bên phải vạch có **3a (132.5 ms)** và **3c (62.5 ms)**. Đoán đúng ở 3a và 3b; trượt ở hai điểm: MLP trên BoW (hàng 2) cũng nằm trên Pareto (p95 chỉ 0.38 ms mà F1 SUB 0.947 > 0.919), và 3c vượt SLA dù tôi nghĩ cả nhóm embedding có độ trễ như nhau. Phần LLM không kiểm chứng được vì không đo.
- *Vì sao lệch:* tôi coi độ trễ của nhóm embedding bị encoder MiniLM quyết định nên sẽ ngang nhau; thực tế p50 của 3b/3c khá gần (31.8 và 37.7 ms) nhưng **p95 của 3c lên 62.5 ms**, tức đuôi độ trễ dao động. Đây chỉ là một lần đo trên Colab nên chưa kết luận được 3c "chậm hơn" 3b về bản chất, nhưng nó cho thấy 3b chỉ cách vạch 50 ms khoảng 9 ms.

*Hai ghi chú thêm:* dự đoán `ml_f1` tôi chỉ chép lại câu hỏi mà không ghi số F1 cụ thể, nên không có gì để so (thực tế: F1 test 0.908, precision 1.000 > recall 0.832). Các dự đoán `llm_models` và `llm_fewshot` không thể đối chiếu vì Phần 4 chưa chạy.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất: BoW + LogReg.** F1 giảm từ 0.919 (SUB) xuống **0.286** trên 50 case khó (−0.633); accuracy còn 50%.

| Model | F1 test thường (SUB) | F1 50 case khó | Mức giảm | Accuracy 50 case khó |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.919 | 0.286 | −0.633 | 50% |
| 2. BoW + MLP | 0.947 | 0.389 | −0.558 | 56% |
| 3a. Emb + kNN | 0.974 | 0.772 | −0.202 | 74% |
| 3b. Emb + LogReg | 0.974 | 0.640 | −0.334 | 64% |
| 3c. Emb + MLP head | 0.974 | 0.627 | −0.347 | 62% |

Độ chính xác theo nhóm (đúng %):

| Nhóm | BoW+LR | BoW+MLP | 3a kNN | 3b | 3c |
| --- | --- | --- | --- | --- | --- |
| Tưởng spam nhưng không phải (10 ham) | 100 | 100 | 70 | 80 | 80 |
| Thao túng / lừa đảo (8 spam) | 12 | 38 | 88 | 38 | 38 |
| Spam giọng tự nhiên (10 spam) | 0 | 0 | 50 | 30 | 30 |
| Né từ khoá (6 spam) | 17 | 0 | 83 | 83 | 83 |
| Prompt injection (5 spam + 1 ham) | 67 | 83 | 100 | 100 | 100 |
| Nói về spam (4 ham) | 100 | 100 | 25 | 50 | 50 |
| Tin giao dịch hợp lệ (6 ham) | 83 | 100 | 100 | 83 | 67 |
| **Tổng (50)** | **50** | **56** | **74** | **64** | **62** |

BoW tụt mạnh nhất ở "spam giọng tự nhiên" (0/10), "thao túng/lừa đảo" (1/8) và "né từ khoá" (1/6). Có **6 case mà cả 5 model đều sai**: #12, #13, #15, #16, #19 (spam giọng tự nhiên) và #22 (giả ngân hàng). Đáng chú ý là thứ hạng **đảo chiều so với test thường**: trên 775 tin test, 3b tốt nhất (F1 0.973) nhưng trên 50 case khó, 3a kNN dẫn đầu (74%).

**Case 1 — #12 (spam giọng tự nhiên, thật = spam, cả 5 model sai):** *"hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club"*.

- BoW: tin này gần như không có từ nào thuộc nhóm đẩy về spam (`call, txt, free, reply, stop, now…`), trong khi `me, my, i, it` là các từ kéo mạnh về ham; mỗi từ là một cột độc lập nên BoW không thể "đọc ra" ý định dụ sang trang hẹn hò.
- Embedding: câu này có nghĩa của một lời nhắn xã giao thân thiện, nên vector nằm gần vùng ham. Tập train kiểu SMS Spam Collection chủ yếu là trúng thưởng, nhạc chuông, số điện thoại tính phí; hầu như không có mẫu "kéo vào trang web bằng giọng người quen", nên mặt phẳng phân loại chưa từng thấy kiểu này. (Đây là giải thích suy ra từ cấu trúc dữ liệu và kết quả, tôi chưa kiểm chứng bằng cách soi các lân cận của tin này.)

**Case 2 — #47 (nói về spam, thật = ham):** *"lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for that?"*

- BoW đoán **đúng** (cả nhóm "nói về spam" BoW đạt 100%), vì `lol`, `i`, `that` kéo về ham đủ để át `won`, `call`; nhưng đây là may mắn do thiên lệch về ham chứ không phải hiểu "kể lại để chê".
- Model 3c (và 3a/3b ở mức 25–50% của cả nhóm) đoán **sai** thành spam: embedding mã hoá *chủ đề* của câu (trúng £1000, số 0906), nên câu **nhắc tới** spam nằm sát vùng spam trên bản đồ nghĩa. Chúng không phân biệt được "dùng" một câu lừa đảo với "bàn về" nó; để làm được cần mô hình đọc ngữ cảnh và ý định, như LLM, nhưng tôi chưa chạy được LLM nên không có số đo để khẳng định LLM làm tốt hơn.

Một quan sát phụ về prompt injection: cả 3 model embedding đạt 100% ở nhóm này, vì chúng không "đọc lệnh" mà chỉ nhìn nội dung (các tin đều chứa nội dung spam rõ như prize, cruise, offer). Dự đoán ban đầu của tôi rằng injection là nhóm khó nhất cho LLM chưa có số liệu để kiểm chứng.

## 5. Thử thách cá nhân

**Đề (theo SEED 8913 → đề #4):** *Embedding · Chạy `knn_predict` với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?*

**Cách làm:** dùng lại `knn_predict(E_tr, y_tr, E_te, k)` (vector MiniLM 384 chiều, cosine bằng nhân ma trận, bỏ phiếu đa số) và `metrics(y_te, preds)` trên toàn bộ 775 tin test, lặp k ∈ {1, 5, 25}.

**Kết quả:**

| k | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| 1 | 0.978 | 0.882 | 0.947 | 0.914 |
| 5 | 0.982 | 0.909 | 0.947 | **0.928** |
| 25 | 0.969 | 0.826 | 0.947 | 0.882 |

Test có 95 spam; recall 0.947 nghĩa là bắt được 90/95 ở cả ba giá trị k (5 FN không đổi). Điều thay đổi là **FP**: tính ngược từ precision, k=1 có ~12 tin ham bị chặn nhầm, k=5 có ~9, k=25 có ~19.

**Giải thích:**

- **k quá nhỏ (k=1):** kết quả phụ thuộc vào đúng một hàng xóm, nên chỉ cần một tin nhiễu, một tin gán nhãn lạ hoặc một mẫu spam quá giống bề mặt là đủ lật dự đoán. Phương sai cao → FP tăng (~12) so với k=5.
- **k vừa (k=5):** bỏ phiếu 5 tin giúp triệt các hàng xóm lẻ lạc, cho F1 tốt nhất (0.928) và ít FP nhất (~9).
- **k quá lớn (k=25):** vùng "hàng xóm" rộng ra khỏi cụm cục bộ, bắt đầu gom cả các tin ở vùng giữa hai lớp; ranh giới trở nên trơn, mất chi tiết cục bộ. Kết quả đo được là FP tăng mạnh (~19) và F1 giảm xuống 0.882. Tôi từng đoán k lớn sẽ làm giảm *recall* vì spam là lớp thiểu số (~12.8% train, cần > 12/25 phiếu); thực tế recall giữ nguyên còn precision giảm. Một lời giải thích hợp lý là spam trong dữ liệu có nhiều biến thể gần giống nhau tạo thành cụm lớn, nên recall không bị ảnh hưởng. Tuy nhiên tôi chưa kiểm tra bằng cách in danh sách tin ham bị chặn nhầm ở k=25 cùng 25 hàng xóm của chúng, nên coi đây là giả thuyết.
- Kết luận: có điểm cân bằng bias–variance quanh k nhỏ-vừa (ở đây k=5); chọn k nên dựa trên tập validation thay vì test.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** p95 < 50 ms/tin, 10 triệu SMS/ngày ≈ 10.000.000 / 86.400 ≈ **116 tin/giây** trung bình (giờ cao điểm cao hơn).

**Số liệu từ bảng của tôi:**

| Phương án | F1 test | F1 50 case khó (acc) | p95 (ms) | Đạt SLA 50 ms? | Chi phí |
| --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.908 | 0.286 (50%) | 0.20 | ✅ | 0 |
| 2. BoW + MLP | 0.940 | 0.389 (56%) | 0.38 | ✅ | 0 |
| 3a. Encoder + kNN | 0.928 | 0.772 (74%) | 132.5 | ❌ | 0 |
| 3b. Encoder + LogReg | **0.973** | 0.640 (64%) | **40.6** | ✅ | 0 |
| 3c. Encoder + MLP head | 0.968 | 0.627 (62%) | 62.5 | ❌ | 0 |
| LLM (gpt-6-luna) | *không đo* | *không đo* | *không đo* | *chưa biết* | ≈ $105/ngày |

**Chi phí LLM (ước tính từ tính tay #5, không phải số đo):** 10M tin × 100 token vào = 1 tỷ token → $100; 10M token ra → $5; tổng ≈ **$105/ngày ≈ $3.150/tháng** (30 ngày) nếu gọi LLM cho *mọi* tin, giả định mỗi tin 100 token vào và 1 token ra với giá $0.10 / $0.50 mỗi 1M token. Độ trễ LLM tôi chưa đo được nên không thể khẳng định LLM có nằm trong p95 < 50 ms hay không; cần đo trước khi đưa vào đường xử lý đồng bộ.

**Phương án chọn:** **3b (MiniLM + LogReg) làm bộ lọc chính** — đây là cách ít lỗi nhất (F1 0.973) trong các cách đạt SLA, chi phí gọi API bằng 0. Kèm theo thiết kế lai (router):

1. Mọi tin đi qua 3b (p95 40.6 ms). Tin có xác suất p ngoài khoảng lưỡng lự (ví dụ p ∈ [0.3, 0.7]) được quyết định ngay.
2. Chỉ phần nhỏ tin lưỡng lự mới chuyển sang LLM, xử lý **bất đồng bộ** (giữ lại hoặc gắn cờ, không chặn đường p95 50 ms). Nếu giả sử khoảng 5% tin vào nhánh này (giả định, chưa đo), chi phí giảm còn ≈ $5/ngày ≈ $160/tháng thay vì $3.150.
3. Thêm bước **chuẩn hoá chống né từ khoá** (gộp `F R E E`, thay `0→o`, `€→e`…) trước khi encode, và đưa các case khó (spam giọng tự nhiên, giả mạo ngân hàng/người thân) vào dữ liệu train định kỳ, vì 3b chỉ đạt 30% ở "spam giọng tự nhiên" và 38% ở "thao túng/lừa đảo".

**Vì sao không chọn các phương án khác:**

- 1 và 2 nhanh nhất (< 0.4 ms) nhưng bỏ sót nhiều: recall 0.832/0.905 và chỉ đúng 50–56% trên 50 case khó (BoW + LogReg chỉ bắt 5/29 spam khó).
- 3a có kết quả tốt nhất trên 50 case khó (74%) nhưng p95 132.5 ms vượt SLA, và với 10 triệu tin, kNN phải lưu và so khớp với toàn bộ vector train nên không mở rộng được nếu không có chỉ mục gần đúng.
- 3c không hơn 3b về F1 (0.968 so với 0.973) mà p95 62.5 ms đã vượt vạch.

**Đánh đổi chấp nhận:**

- Chấp nhận 3b vẫn bỏ sót khoảng 4% spam trên test thường (recall 0.958) và yếu trên các case khó (64%); nhánh LLM và dữ liệu mới là thứ để bù, nhưng hiệu quả nhánh LLM **chưa được kiểm chứng** trong bài này.
- p95 40.6 ms chỉ cách ngưỡng ~9 ms và đo trên một lần chạy Colab, gọi tuần tự từng tin, chưa tính độ trễ mạng và tải đồng thời. Cần đo lại trên phần cứng thật.
- **Throughput:** 3b đạt ~44 tin/s cho mỗi tiến trình (đo ở chế độ cả lô), trong khi tải trung bình là ~116 tin/s, nên cần ít nhất ~3 bản sao (nhiều hơn nữa cho giờ cao điểm), hoặc dùng GPU / batching. Đây là chi phí hạ tầng thay cho chi phí API.
- Bộ test sinh ra từ cùng phân phối với train nên điểm F1 test hơi lạc quan (vẫn còn các spam mẫu gần giống nhau ở cả hai tập sau khi chỉ bỏ trùng tuyệt đối); cần theo dõi chất lượng bằng dữ liệu thật sau khi triển khai.

## 7. AI log và điều đã học

> **Lưu ý trước khi nộp:** bảng dưới là bản nháp dựa trên dấu vết trong notebook. Hãy chỉnh lại cho khớp đúng những gì bạn thực sự đã hỏi/nhận từ AI; xoá hoặc thêm dòng nếu cần. Phần 1–3 là vùng tự code nên tôi không ghi nội dung dùng AI để viết code ở đó.

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4 — TODO 4.1 `call_openai` | Viết `call_openai` theo template: gọi Chat Completions, trả về `(text, latency)` | Hàm dùng `client = OpenAI()`, đo `time.perf_counter()`, truyền `**provider_kwargs(max_tokens)` | Ô kiểm tra báo `NameError: OpenAI`: chưa import `OpenAI` trong hàm; cần thêm `from openai import OpenAI` bên trong hàm rồi chạy lại check 4.1 + 4.4 |
| Phần 4 — TODO 4.2–4.3 | Viết `build_classification_prompt` (zero-shot / few-shot) và `parse_label` | Prompt kết thúc bằng `Message: """..."""\nLabel:`; `parse_label` xử lý chữ hoa, dấu câu, `Label: spam`, "not spam" | Check 4.2–4.3 đạt ✅ (kiểm tra các trường hợp ` Spam.`, `HAM\n`, `Not spam`, `I am not sure` → None) |
| Phần 4 — TODO 4.4 `llm_classify_batch` | Viết hàm phân loại cả loạt, đo latency, ước tính chi phí, đếm kết quả không hợp lệ | Hàm lặp từng tin, dùng `retry_with_backoff`, ước tính token = số từ / 0.75, chi phí tính riêng input/output theo `PRICE_PER_1M` | Chưa kiểm chứng được vì check bị lỗi `OpenAI` và không có API key để chạy thật |
| Phần 5 — Câu hỏi 5, 6 | Giúp trả lời các câu hỏi phân tích biểu đồ Pareto và stress test | Bản trả lời gồm các nhận định về Pareto, injection, chuẩn hoá chống né từ khoá | Đối chiếu lại với số liệu thật: bản nháp cho rằng BoW khó nhất ở "né từ khoá" và embedding/LLM khó nhất ở "prompt injection". Số đo cho thấy BoW 0% ở "spam giọng tự nhiên" (né từ khoá 17%) và embedding đạt 100% ở prompt injection; LLM chưa chạy. Báo cáo này dùng số liệu thật thay cho các nhận định chưa kiểm chứng |

**Điều tôi hiểu ra hôm nay mà trước đó chưa hiểu:**

Trước hôm nay tôi nghĩ F1 trên tập test cao (0.91–0.97) nghĩa là mô hình "hiểu" spam. Stress test cho thấy điều đó sai: cùng một BoW + LogReg từ 0.919 rơi xuống 0.286, và thứ hạng đảo chiều (3b tốt nhất trên test thường nhưng kNN dẫn đầu trên 50 case khó). Tôi cũng nhận ra con số đẹp như "100% ở nhóm ham" có thể chỉ phản ánh việc mô hình luôn đoán ham; muốn biết mô hình giỏi thật phải xem precision và recall từng nhóm.

Thứ hai, chọn mô hình cho sản phẩm không chỉ là chọn F1 cao nhất mà là trả lời đồng thời: p95 có dưới 50 ms không, bao nhiêu bản sao để chịu 116 tin/giây, và chi phí mỗi tháng (LLM cho mọi tin ≈ $3.150). Thứ ba, tôi dự đoán sai nhiều nhất ở chỗ đánh giá quá cao khả năng "hiểu nghĩa" của embedding và đánh giá thấp việc BoW thiên về đoán ham; và bài học thực hành là phải ghi dự đoán có nội dung cụ thể (mô hình nào, con số nào) thì mới đối chiếu được sau khi chạy.
