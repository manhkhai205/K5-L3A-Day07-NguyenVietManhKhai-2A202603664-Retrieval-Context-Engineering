# Code Day 7 — Agentic RAG

Thư mục này chứa phần cài đặt kết nối dữ liệu Day 6, các phương pháp retrieval, vòng lặp agent dùng Gemini thật và script đánh giá.

## Cấu trúc file

- `prepare_day7_data.py`: Xuất 362 chunks chiến lược B từ Day 6 Qdrant sang `data/chunks.json`, tạo bộ benchmark `data/questions.json`, `data/unanswerable_questions.json`, `data/cases.json` (22 `dev` + 7 `test`) và kiểm chứng kết nối `gemini-embedding-2` + Qdrant.
- `retrievers.py`: 4 phương pháp retrieval (`TfidfRetriever`, `BM25Retriever`, `DenseQdrantRetriever`, `HybridRRFRetriever`), hàm suy luận filter tất định (`infer_filters_from_query`) và đóng gói context (`pack_context_smart`).
- `benchmark_retrieval.py`: Chạy so sánh các phương pháp retrieval và cấu hình context budget trên `dev`/`test`.
- `agent_runner.py`: `GeminiController` (gọi `gemini-3.8-flash` structured JSON output, xử lý retry khi `answer` rỗng) và `run_agent_smart` (xen kẽ ứng viên đa lượt và mở rộng chunk lân cận).
- `benchmark_agent.py`: Chạy agent thật trên `dev`/`test`, lưu đầy đủ trace, token, thời gian và thống kê.

## Hướng dẫn chạy tái lập

Chạy từ thư mục gốc với môi trường Python 3.11+ đã cài `google-genai`, `qdrant-client`, `python-dotenv`, `pydantic` và đã cấu hình `.env` (từ `.env.example`):

```bash
# 1. Chạy toàn bộ unit tests (21 tests)
python -m unittest discover -s tests -v

# 2. (Tùy chọn nếu muốn tái xuất từ Qdrant Day 6) Chuẩn bị dữ liệu & kiểm tra kết nối
python src/prepare_day7_data.py

# 3. Benchmark retrieval trên dev (7 cấu hình) và test (Baseline + Chọn)
python src/benchmark_retrieval.py --split dev
python src/benchmark_retrieval.py --split test --configs baseline_tfidf_b800 improve3_hybrid_rrf_inferred_b3200

# 4. Chạy Agent thật trên dev và test
python src/benchmark_agent.py --split dev
python src/benchmark_agent.py --split test --configs agent_baseline_tfidf_b800 agent_improve2_hybrid_smart_b3200

# 5. Tổng hợp nhãn audit thủ công trên test bằng starter/agent_metrics.py
python starter/agent_metrics.py --cases data/cases.json --traces results/agent/agent_baseline_tfidf_b800_test/traces.json --reviews results/agent/agent_baseline_tfidf_b800_test/reviews.json --split test --output results/agent/agent_baseline_tfidf_b800_test/summary.json
python starter/agent_metrics.py --cases data/cases.json --traces results/agent/agent_improve2_hybrid_smart_b3200_test/traces.json --reviews results/agent/agent_improve2_hybrid_smart_b3200_test/reviews.json --split test --output results/agent/agent_improve2_hybrid_smart_b3200_test/summary.json
```
