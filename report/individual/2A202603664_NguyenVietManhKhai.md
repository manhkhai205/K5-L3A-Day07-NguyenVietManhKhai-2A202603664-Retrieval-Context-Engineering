# Báo cáo Day 7 — Agentic RAG & Retrieval / Context Engineering

- **Học viên**: Nguyễn Viết Mạnh Khải — MSSV: `2A202603664` — Lớp: `K5-L3A`
- **Ghi nhận sử dụng AI & kiểm chứng**: Có sử dụng trợ lý AI hỗ trợ lập trình pipeline (`src/`), viết unit test và nháp cấu trúc báo cáo; toàn bộ thực nghiệm đều chạy trực tiếp trên kho chunk thật từ Day 6 (Qdrant + `gemini-embedding-2` 768 chiều) và model thật (`gemini-3.8-flash`), toàn bộ câu trả lời và trích dẫn (`[chunk-id]`) trên tập `test` đều được đọc đối chiếu thủ công với văn bản nguồn chuẩn hóa trước khi ghi nhãn vào `reviews.json`.

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk**:
  - Tái sử dụng trực tiếp bộ 2 tài liệu công cộng của NIST từ Day 6 (miễn phí, thuộc phạm vi công cộng của Bộ Thương mại Hoa Kỳ):
    1. `nist_ai_100_1` — *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, Jan 2023 (`https://doi.org/10.6028/NIST.AI.100-1`), `42` trang nội dung (trang PDF `6–47`).
    2. `nist_ai_600_1` — *AI RMF: Generative Artificial Intelligence Profile*, Jul 2024 (`https://doi.org/10.6028/NIST.AI.600-1`), `53` trang nội dung (trang PDF `5–57`).
  - Tổng cộng `95` trang nội dung, **362 chunks** từ Chiến lược B (`by_paragraph`, `max_size=900` ký tự, trung bình `652.1` ký tự/chunk), xuất trực tiếp từ collection `day6_nist_chunks` (`strategy="B"`) sang [data/chunks.json](../../data/chunks.json) thông qua [src/prepare_day7_data.py](../../src/prepare_day7_data.py) và [starter/convert_day6.py](../../starter/convert_day6.py).
- **Vector store, embedding model, phiên bản và số chiều**:
  - **Vector store**: `Qdrant` (`qdrant-client==1.20.0`, Docker image `qdrant/qdrant:v1.20.0` tại `http://127.0.0.1:6333`, collection `day6_nist_chunks`, độ đo `COSINE`, kèm cơ chế lưu cục bộ bền vững `QdrantClient(path=...)` trong [src/retrievers.py](../../src/retrievers.py) phòng trường hợp Docker daemon tắt nền trên Windows).
  - **Embedding model**: Google Gemini API (`google-genai==3.3.0`), `gemini-embedding-2`, cố định **`768` chiều** (`task_type="RETRIEVAL_DOCUMENT"` khi index, `"RETRIEVAL_QUERY"` khi truy vấn). Kiểm tra tương thích live trên `q01` trả về Top-1 `nist_ai_100_1:B:p06:c002` với score `0.869812`.
- **Agent model, cách chạy và phân chia dev/test**:
  - **Agent LLM**: `gemini-3.8-flash` (do `gemini-2.5-flash` trả về HTTP `404 NOT_FOUND` cho tài khoản mới trong lần thử đầu tiên — xem trace lỗi lưu tại [results/agent/agent_improve2_hybrid_smart_b3200_dev/traces.json](../../results/agent/agent_improve2_hybrid_smart_b3200_dev/traces.json) trước khi chuyển sang `gemini-3.8-flash`), `temperature=0.0`, `thinking_budget=0`, structured JSON output (`AgentDecisionSchema`).
  - **Phân chia `dev` / `test` ([data/cases.json](../../data/cases.json) — tổng `29` câu)**:
    - **`dev` (`22` câu)**: `20` câu có đáp án (`q01–q20`, trong đó `q16–q20` của Day 6 được chuyển sang `dev` vì đã được đọc phân tích lỗi trong báo cáo Day 6, tuân thủ quy tắc không dùng lại câu đã xem để chấm `test`) + `2` câu thiếu bằng chứng (`u01–u02`).
    - **`test` (`7` câu mới hoàn toàn, khóa trước khi chạy)**: `5` câu có đáp án (`q21–q25`, phủ các dạng trực tiếp, điều kiện/ngoại lệ, diễn đạt lại và metadata filter) + `2` câu thiếu bằng chứng (`u03–u04`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm**:
  - Với câu có đáp án (`q01–q25`), toàn bộ `gold` (`source_id`, `page`, `evidence`) đều vượt qua kiểm tra `validate()` khớp nguyên văn trên một chunk (riêng `q06` có 2 nguồn hợp lệ `100-1 p.35` và `600-1 p.36`).
  - Với câu thiếu bằng chứng (`u01–u04`), câu hỏi sát chủ đề NIST AI RMF (tiền phạt % doanh thu, ngưỡng $10^{26}$ FLOPs đăng ký liên bang, biểu phí USD chứng chỉ tuân thủ NIST, điểm sàn MMLU/GSM8K cho y tế) nhưng hàm `verify_unanswerable_against_corpus()` quét **toàn bộ 362 chunks** để xác nhận 0 lần xuất hiện.
  - **Tiêu chí chốt cấu hình trên `dev`**: Ưu tiên cao nhất cho `Context Evidence Rate` & số câu trả lời đúng có căn cứ (`answered_on_answerable` với `0` false abstention), tiếp đến là `Evidence Hit@3`, khả năng từ chối đúng câu thiếu bằng chứng (`Abstention`), và đánh đổi độ trễ/số lượt LLM calls.

---

## 2. Retrieval và thử nghiệm

Đã triển khai **4 phương pháp retrieval khác nhau** trong [src/retrievers.py](../../src/retrievers.py) (`TfidfRetriever`, `BM25Retriever`, `DenseQdrantRetriever`, `HybridRRFRetriever`) và đo trên cùng 20 câu có đáp án của `dev` qua [src/benchmark_retrieval.py](../../src/benchmark_retrieval.py). Chi tiết lưu tại [results/retrieval/dev_comparison.json](../../results/retrieval/dev_comparison.json) và [results/retrieval/test_comparison.json](../../results/retrieval/test_comparison.json).

| Cấu hình | Phương pháp, top-k, filter, context budget | Giả thuyết / thay đổi | Split | Hit@1 (`n/N`) | Evidence Hit@3 (`n/N`) | Context Evidence Rate (`n/N`) | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| **Baseline** (`baseline_tfidf_b800`) | TF-IDF, `top_k=3`, không filter, `budget=800` | Baseline lexical chuẩn từ starter | `dev` | `14/20` (`70.0%`) | `17/20` (`85.0%`) | `3/20` (`15.0%`) | Median `1.00 ms` (`0 USD`) |
| **Phương pháp 2** (`method2_bm25_b800`) | BM25 + domain code, provided filter, `budget=800` | Chuẩn hóa độ dài đoạn và giữ mã hành động (`MS-2.5`, `MG-3.1`) | `dev` | `14/20` (`70.0%`) | `17/20` (`85.0%`) | `3/20` (`15.0%`) | Median `1.32 ms` (`0 USD`) |
| **Phương pháp 3** (`method3_dense_qdrant_b800`) | Dense Qdrant Day 6 (`gemini-embedding-2`), provided filter, `budget=800` | Bắt ngữ nghĩa câu hỏi diễn đạt lại tốt hơn lexical | `dev` | `17/20` (`85.0%`) | `18/20` (`90.0%`) | `4/20` (`20.0%`) | Median `14.89 ms`* (`0 USD`, free tier) |
| **Phương pháp 4** (`method4_hybrid_rrf_b800`) | Hybrid Dense+BM25 RRF (`k=60`, `w_d=1.35`), provided filter, `budget=800` | Kết hợp thứ hạng ngữ nghĩa và từ khóa hiếm | `dev` | `15/20` (`75.0%`) | `18/20` (`90.0%`) | `3/20` (`15.0%`) | Median `31.63 ms`* (`0 USD`) |
| **Cải thiện 1** (`improve1_dense_qdrant_b2400`) | Dense Qdrant + provided filter + **`budget=2400`** | Nâng budget để không cắt rơi chunk Chiến lược B (`650–900` ký tự) | `dev` | `17/20` (`85.0%`) | `18/20` (`90.0%`) | `18/20` (`90.0%`) | Median `22.34 ms`* (`0 USD`) |
| **Cải thiện 2** (`improve2_hybrid_rrf_b2400`) | Hybrid RRF + provided filter + **`budget=2400`** | Phối hợp Hybrid RRF với `budget=2400` | `dev` | `15/20` (`75.0%`) | `18/20` (`90.0%`) | `17/20` (`85.0%`) | Median `21.87 ms`* (`0 USD`) |
| **Cải thiện 3 / Chọn** (`improve3_hybrid_rrf_inferred_b3200`) | Hybrid RRF + **Inferred Filter** + **Neighbor Expand**, `budget=3200` | Tự suy luận `year`/`section` tất định + ghép chunk lân cận cùng trang | `dev` | `16/20` (`80.0%`) | `19/20` (`95.0%`) | `19/20` (`95.0%`) | Median `22.19 ms`* (`0 USD`) |
| **Cấu hình đã chọn** (`improve3_hybrid_rrf_inferred_b3200`) | Hybrid RRF + Inferred Filter + Neighbor Expand, `budget=3200` | Đánh giá độc lập sau khi chốt | `test` | `3/5` (`60.0%`) | `5/5` (`100.0%`) | `5/5` (`100.0%`) | Median `531.69 ms`** (`0 USD`, 5 embed calls) |
| **Baseline** (`baseline_tfidf_b800`) | TF-IDF, `top_k=3`, không filter, `budget=800` | Đánh giá đối chiếu trên test | `test` | `5/5` (`100.0%`) | `5/5` (`100.0%`) | `0/5` (`0.0%`) | Median `3.37 ms` (`0 USD`) |

*\*Đo khi tái dùng vector query đã cache; \*\*Bao gồm gọi trực tiếp API `gemini-embedding-2` (5 query test mới).*

### Phân tích 2 lỗi dev, thử nghiệm không tăng điểm và lý do chốt cấu hình
1. **Lỗi `dev` 1 — Nghẽn `budget=800` ký tự trên `q01`, `q05`, `q06`... (Baseline & các cấu hình `b800`)**:
   - *Nguyên nhân*: `pack_context` mặc định chỉ cho phép `budget=800` ký tự (gồm cả header `[chunk_id | source | page X]\n` ~78 ký tự), trong khi phần lớn chunk Chiến lược B dài `730–900` ký tự. Dù Dense Qdrant `b800` đạt `Hit@3 = 18/20 (90%)`, có tới 14 câu bị loại sạch chunk đúng khỏi context nên `Context Evidence Rate` chỉ còn `4/20 (20%)`.
   - *Thay đổi & kết quả*: Khi tăng `budget` lên `2400` (`improve1_dense_qdrant_b2400`), `Context Evidence Rate` nhảy vọt từ **`20%` (`4/20`) lên `90%` (`18/20`)**.
2. **Lỗi `dev` 2 — `q02` (rơi vế định nghĩa sang chunk kề) và `q19` (trượt từ khóa sang `100-1`)**:
   - *Nguyên nhân*: Ở `q02`, `nist_ai_100_1:B:p12:c002` đứng Rank 1 nhưng câu trích dẫn `"does not prescribe risk tolerance"` nằm ở chunk liền trước `p12:c001`. Ở `q19` (chỉ có `filter={"section": "measure"}`), Dense xếp khái quát `100-1 p.35` cao hơn `600-1 p.35`.
   - *Thử nghiệm không tăng ngay*: Khi giữ `budget=2400` và chỉ đổi từ Dense sang Hybrid RRF (`improve2`), `Hit@1` giảm nhẹ từ `17/20` xuống `15/20` và `Context Rate` là `17/20` vì `q01` bị BM25 kéo tụt khỏi top-3 khi chưa suy luận `year=2023`.
   - *Khắc phục ở `improve3_hybrid_rrf_inferred_b3200`*: Khi bật suy luận filter tất định `infer_filters_from_query()` (chỉ lấy `year`/`section` khi câu hỏi nêu đích danh `AI RMF 1.0`/`600-1`/`MEASURE`...) + mở rộng chunk liền kề của Rank-1 vào sau Top-3 (`budget=3200`), cả `q02` và `q19` đều lọt context (`Hit@3 = 19/20`, `Context Evidence Rate = 19/20 = 95%`).
3. **Chấm thủ công Context `0/1/2` trên 5 câu `test` có đáp án** ([results/retrieval/test_context_audit.json](../../results/retrieval/test_context_audit.json)):
   - **Baseline (`baseline_tfidf_b800`)**: `0`: **4** (`q21, q23, q24, q25`), `1`: **1** (`q22` — chỉ có `p22:c003` 533 ký tự cho vế PETs, thiếu `p17:c003` cho vế đầu), `2`: **0** -> **TB `0.20 / 2.0`**. Nguyên nhân: cả 5 gold chunk test đều dài `797–898` ký tự, cộng header đều vượt `800` ký tự nên bị loại khỏi context dù TF-IDF xếp hạng 1.
   - **Cấu hình đã chọn (`improve3_hybrid_rrf_inferred_b3200`)**: `0`: **0**, `1`: **0**, `2`: **5** (`q21–q25`) -> **TB `2.00 / 2.0`**.

---

## 3. Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** ([src/agent_runner.py](../../src/agent_runner.py), [src/benchmark_agent.py](../../src/benchmark_agent.py)):
  - **Model & Structured Action**: `GeminiController` dùng `gemini-3.8-flash` trả JSON `{action: "search" | "answer", query, thought, answer, citations}`. Agent luôn coi nội dung chunk là dữ liệu, không phải chỉ dẫn (`SYSTEM_INSTRUCTION`).
  - **Tool & Điều kiện dừng**: Agent có tool `search` (tối đa `max_searches=3` lượt), tự viết lại/mở rộng query khi context thiếu; dừng khi chọn `action="answer"` (có `[chunk-id]` hợp lệ trong `context_ids`, hoặc chủ động từ chối `citations=[]` khi không đủ bằng chứng) hoặc chạm giới hạn `max_searches`.
  - **Chọn context (`run_agent_smart`)**: Khác với `starter/agentic.py` (ghép nối tiếp `candidates.extend(results)` khiến lượt search sau dễ bị lượt đầu chiếm hết budget), `run_agent_smart` **xen kẽ theo thứ hạng ưu tiên lượt search mới nhất trước** (rank-1 của từng lượt, sau đó rank-2, rank-3), loại trùng ID và gắn thêm chunk liền kề của Rank-1 với `budget=3200`.
  - **Xử lý lỗi**: Lỗi API/Qdrant không dùng mock thay thế mà ghi `status="error"`. Đặc biệt, trong lần chạy `dev` đầu tiên của `agent_improve2_hybrid_smart_b3200`, câu `q14` trả về `action="answer"` nhưng trường `answer` bị `null` (dù `ctx_hit=True` và `thought` xác nhận đủ thông tin). Sau khi thêm cơ chế tự nhắc/retry khi `answer` rỗng trong `GeminiController.decide`, `q14` trên `dev` đã trả lời đầy đủ và chính xác (`20/20` answered, `0` false abstention, `2/2` abstention trên `dev`).

### Bảng tổng hợp Agent trên `dev` và `test`

*(Kết quả `test` được tổng hợp bằng [starter/agent_metrics.py](../../starter/agent_metrics.py) từ các nhãn đọc đối chiếu thủ công trong `reviews.json`)*:

| Cấu hình / Split | Tool use | Số lượt gọi (TB / phân bố) | Grounded answer `0/1/2` và TB | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Agent Baseline** (`agent_baseline_tfidf_b800`, `dev`) | `22/22` | `2.68` (`1`:3, `2`:1, `3`:18) | *(chạy đối chiếu dev: `8/20` answered)* | — | — | `2/2` (`100%`) | `12/20` (`60%`) |
| **Agent Cải thiện 1** (`agent_improve1_dense_b2400`, `dev`) | `22/22` | `1.32` (`1`:18, `2`:1, `3`:3) | *(chạy đối chiếu dev: `19/20` answered; trượt `q12`)* | — | — | `2/2` (`100%`) | `1/20` (`5%`) |
| **Agent Chọn** (`agent_improve2_hybrid_smart_b3200`, `dev`) | `22/22` | `1.09` (`1`:21, `3`:1) | *(chạy đối chiếu dev: `20/20` answered)* | — | — | `2/2` (`100%`) | `0/20` (`0%`) |
| **Agent Baseline** (`agent_baseline_tfidf_b800`, `test`) | `7/7` (`100%`) | `2.71` (`2`:2, `3`:5) | `0`:4, `1`:1, `2`:0 \| **TB `0.20 / 2.0`** | `1/1` (`100%`) | `1/1` (`100%`) | `2/2` (`100%`) | `4/5` (`80%`) |
| **Agent Chọn** (`agent_improve2_hybrid_smart_b3200`, `test`) | `7/7` (`100%`) | `1.00` (`1`:7) | `0`:0, `1`:0, `2`:5 \| **TB `2.00 / 2.0`** | `7/7` (`100%`) | `6/6` (`100%`) | `1/2` (`50%`)* | `0/5` (`0%`) |

*\*Giải thích trường hợp `u03` ở `test`: về mặt ngữ nghĩa, Agent đã **từ chối rõ ràng** câu hỏi về biểu phí USD và thời hạn chứng chỉ (`"do not contain evidence of an official fee schedule, certificate validity period..."`) nhưng trích dẫn thêm `[nist_ai_100_1:B:p07:c003]` để giải thích AI RMF là khung tự nguyện. Do có `citations` khác rỗng, vòng lặp starter gán `status="answered"` thay vì `"abstain"`, nên [starter/agent_metrics.py](../../starter/agent_metrics.py) bắt buộc ghi `abstention_correct=false` theo quy tắc kiểm tra khớp `status` (`1/2` theo status, dù thực chất hành vi từ chối đúng `2/2`).*

### Chi tiết kiểm chứng từng câu `test` (Cấu hình đã chọn `agent_improve2_hybrid_smart_b3200`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q21` | Có đáp án (`truc_tiep`) | 1 lượt `search` (`filter={"year": 2023}`), context 3 chunks gồm `p17:c002` | Giải thích `Valid & Reliable` là điều kiện nền tảng và `Accountable & Transparent` là hộp dọc liên kết mọi đặc tính `[nist_ai_100_1:B:p17:c002]`. | **Grounded = 2**, citation `[True]`. Đối chiếu đúng nguyên văn `p.17`. |
| `q22` | Có đáp án (`dieu_kien_ngoai_le`) | 1 lượt `search` (`filter=None`), context gồm `p17:c003` và `p22:c003` | Nêu đủ đánh đổi/tính yếu nhất `[nist_ai_100_1:B:p17:c003]` và suy giảm độ chính xác của PETs khi dữ liệu thưa `[nist_ai_100_1:B:p17:c003, nist_ai_100_1:B:p22:c003]`. | **Grounded = 2**, citation `[True, True]`. Cả 2 nguồn hỗ trợ đầy đủ 2 vế. |
| `q23` | Có đáp án (`dien_dat_lai`) | 1 lượt `search` (`filter={"year": 2024}`), context có `p15:c002` + lân cận | Phân biệt đúng direct prompt injection (nhập trực tiếp) vs indirect prompt injection (tấn công từ xa qua dữ liệu được truy xuất để đánh cắp dữ liệu/chạy mã độc) `[nist_ai_600_1:B:p15:c002]`. | **Grounded = 2**, citation `[True]`. Đối chiếu khớp hoàn toàn `600-1 p.15`. |
| `q24` | Có đáp án (`metadata_filter`) | 1 lượt `search` (`filter={"year": 2024, "section": "map"}`), RRF kéo `p29:c001` từ rank 6 Dense lên rank 3 | Nêu chính xác cả `MP-2.3-003` (fact-checking) và `MP-2.3-004` (kiểm thử nhận diện nội dung tổng hợp) `[nist_ai_600_1:B:p29:c001]`. | **Grounded = 2**, citation `[True]`. Đối chiếu đúng `600-1 p.29`. |
| `q25` | Có đáp án (`metadata_filter`) | 1 lượt `search` (`filter={"section": "measure"}`), `p42:c001` đạt rank 1 | Nêu đủ `MS-2.13-001` (mô hình sai số đo lường, đo thiên lệch/phương sai và chuyên gia miền) `[nist_ai_600_1:B:p42:c001]`. | **Grounded = 2**, citation `[True]`. Đối chiếu đúng `600-1 p.42`. |
| `u03` | Thiếu bằng chứng | 1 lượt `search` (`filter={"year": 2023}`), context có `p07:c003` | Nêu rõ tài liệu không có thông tin về biểu phí hay thời hạn chứng chỉ tuân thủ; khẳng định AI RMF mang tính tự nguyện `[nist_ai_100_1:B:p07:c003]`. | **Citation `[True]`**, nhưng `status="answered"` nên `abstention_correct=False` theo script tổng hợp. |
| `u04` | Thiếu bằng chứng | 1 lượt `search` (`filter={"year": 2024}`), context có `p53:c002`, `p40:c001`, `p40:c004` | Từ chối dứt khoát, nêu tài liệu không quy định điểm sàn MMLU/GSM8K cho y tế; `citations=[]`. | **`status="abstain"`**, `abstention_correct=True`. Không bịa đặt số liệu. |

### Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí
- **Trace & nhãn**:
  - Retrieval: [results/retrieval/dev_comparison.json](../../results/retrieval/dev_comparison.json), [results/retrieval/test_comparison.json](../../results/retrieval/test_comparison.json), [results/retrieval/test_context_audit.json](../../results/retrieval/test_context_audit.json).
  - Agent `dev`: `results/agent/agent_*_dev/{traces.json,run_stats.json}`.
  - Agent `test`: [results/agent/agent_baseline_tfidf_b800_test/summary.json](../../results/agent/agent_baseline_tfidf_b800_test/summary.json) & [results/agent/agent_improve2_hybrid_smart_b3200_test/summary.json](../../results/agent/agent_improve2_hybrid_smart_b3200_test/summary.json) (kèm `traces.json`, `reviews.json`, `run_stats.json`).
- **Lệnh tái lập chính**:
  - Kiểm tra unit tests: `python -m unittest discover -s tests -v` (`21/21` PASS).
  - Bước 1: `python src/prepare_day7_data.py`
  - Bước 2 (Retrieval `dev` & `test`): `python src/benchmark_retrieval.py --split dev` và `python src/benchmark_retrieval.py --split test --configs baseline_tfidf_b800 improve3_hybrid_rrf_inferred_b3200`
  - Bước 3 (Agent `dev` & `test`): `python src/benchmark_agent.py --split dev` và `python src/benchmark_agent.py --split test --configs agent_baseline_tfidf_b800 agent_improve2_hybrid_smart_b3200`
  - Tổng hợp nhãn test: `python starter/agent_metrics.py --cases data/cases.json --traces results/agent/agent_improve2_hybrid_smart_b3200_test/traces.json --reviews results/agent/agent_improve2_hybrid_smart_b3200_test/reviews.json --split test --output results/agent/agent_improve2_hybrid_smart_b3200_test/summary.json`
- **Thời gian & chi phí trên `test` (`7` câu)**:
  - **Baseline**: `87.61s` (`26` LLM calls, `16.491` prompt tokens + `2.817` completion tokens), chi phí `0 USD` (Free tier).
  - **Cấu hình đã chọn**: `55.23s` (`14` LLM calls, `12.458` prompt tokens + `2.408` completion tokens), chi phí `0 USD` (Free tier). Nhanh hơn Baseline **37%** và giảm **46%** số lần gọi LLM nhờ tìm trúng ngay ở lượt 1 (`1.00` vs `2.71` lượt search/câu).

### Kết quả trước/sau, đánh đổi và giới hạn
1. **Hiệu quả trước/sau**: Cấu hình đã chọn cải thiện điểm `Grounded Answer` trên `test` từ **`0.20 / 2.0`** (`Baseline`) lên **`2.00 / 2.0`** (`5/5` câu đạt điểm tối đa), giảm tỷ lệ từ chối nhầm từ `4/5` xuống `0/5`, đồng thời giảm thời gian chạy trên `test` từ `87.61s` xuống `55.23s`.
2. **Đánh đổi**:
   - Tăng `budget` từ `800` lên `3200` ký tự làm tăng số prompt token mỗi lượt gọi LLM (~`890` vs ~`634` token/câu ở Baseline), nhưng đổi lại giảm mạnh số lượt phải search lại (từ `2.71` xuống `1.00`), nên **tổng prompt token trên toàn tập `test` thực tế còn thấp hơn Baseline** (`12.458` vs `16.491` tokens).
   - `Hybrid RRF` ưu tiên độ phủ (`Hit@3 = 100%`) và kéo được ca khó như `q24` từ rank 6 Dense lên top-3, nhưng ở câu `q21` trọng số RRF khiến `p08:c001` nhỉnh hơn `p17:c002` một biên độ nhỏ (`0.03826` vs `0.03817`), làm `Hit@1` trên test là `3/5` (dù `Hit@3` vẫn đạt `5/5`).
3. **Giới hạn**:
   - Tập `test` gồm `7` câu (`5` có đáp án + `2` thiếu bằng chứng) nên mỗi câu ảnh hưởng lớn đến tỷ lệ phần trăm (ví dụ `u03` khiến metric abstention theo status còn `1/2`).
   - Trường hợp `u03` cho thấy cần phân tách rõ trong schema đầu ra giữa *"từ chối trả lời kèm trích dẫn ngữ cảnh chứng minh tính tự nguyện"* (`status="abstain"`, `evidence_citations=[...]`) và *"trả lời trực tiếp câu hỏi"* (`status="answered"`), tránh việc `agent_metrics.py` phạt nhầm khi Agent trích dẫn nguồn giải thích lý do tài liệu không quy định.
   - Bộ suy luận filter tất định `infer_filters_from_query()` chỉ hiệu quả khi câu hỏi nêu tên khung chuẩn/chức năng viết hoa (`GOVERN/MAP/MEASURE/MANAGE`), chưa xử lý câu hỏi ẩn ý không nêu tên phần.
