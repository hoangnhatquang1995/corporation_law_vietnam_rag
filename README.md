# Corporation Law Vietnam RAG

Ứng dụng hỏi đáp về Luật Doanh nghiệp Việt Nam sử dụng Retrieval-Augmented Generation (RAG). Hệ thống kết hợp FastAPI, Gradio, LangGraph, Qdrant và các mô hình Hugging Face để truy xuất tài liệu pháp lý, sắp xếp lại kết quả và sinh câu trả lời ngắn gọn.

## Tổng quan

- Giao diện web dùng FastAPI và Gradio.
- Agent hỏi đáp dùng LangGraph để phân loại câu hỏi và điều hướng luồng xử lý.
- Vector store mặc định dùng Qdrant.
- Dataset nguồn dùng Hugging Face dataset `vohuutridung/vietnamese-legal-documents`.
- Embedding mặc định dùng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- LLM hiện tại được gọi qua DeepSeek endpoint trong code.

## Tính năng chính

- Trả lời câu hỏi về Luật Doanh nghiệp Việt Nam.
- Tách câu hỏi thành nhóm `legal_question`, `general_knowledge`, hoặc `chitchat`.
- Truy xuất tài liệu liên quan từ vector store.
- Rerank kết quả truy xuất trước khi sinh câu trả lời.
- Hiển thị giao diện chat tại `/` và giao diện Gradio độc lập tại `/gradio`.
- Tự động tạo các thư mục cache cho dataset và model trong repo.

## Kiến trúc nhanh

- `src/main.py`: entrypoint chạy Uvicorn.
- `src/api/api.py`: khởi tạo FastAPI, HTML shell và mount Gradio.
- `src/rag/chatbot/chatbot.py`: nhận câu hỏi và gọi graph.
- `src/rag/agent/`: các node, state và graph của agent.
- `src/dataset/`: tải dataset, chunk văn bản và làm việc với vector store.
- `src/settings/settings.py`: biến cấu hình, đường dẫn cache và tham số hệ thống.

## Yêu cầu

- Python 3.12.
- Qdrant đang chạy và truy cập được.
- API key cho provider LLM đang dùng. Với cấu hình mặc định hiện tại, cần `DEEPSEEK_API_KEY`.
- Kết nối Internet cho lần đầu tải model và dataset.

## Cài đặt local

### 1. Tạo môi trường ảo và cài thư viện

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu `requirements.txt` được tạo bằng PowerShell `pip freeze > requirements.txt`, file có thể đang UTF-16. Dockerfile đã xử lý trường hợp này khi build container. Nếu bạn gặp lỗi đọc file requirements trên một môi trường khác, hãy chuyển file sang UTF-8 trước khi cài.

### 2. Tạo file `.env`

Tạo file `.env` tại thư mục gốc dự án:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
APP_HOST=127.0.0.1
APP_PORT=8000
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_TIMEOUT=120
QDRANT_BATCH_SIZE=16
QDRANT_UPSERT_WAIT=false
```

Code hiện có sẵn đường dẫn cho `OPENAI_API_KEY` và `GOOGLE_API_KEY`, nhưng provider mặc định trong repo đang là DeepSeek.

### 3. Chạy Qdrant

Bạn có thể chạy nhanh Qdrant bằng Docker:

```bash
docker run --rm -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Nếu Qdrant không chạy ở máy local, hãy cập nhật `QDRANT_HOST` và `QDRANT_PORT` cho phù hợp.

### 4. Nạp dữ liệu vào vector store

Repo này không tự động ingest tài liệu vào Qdrant khi khởi động app. Lúc import, `documentDB.build()` chỉ tạo hoặc mở collection, không tự động thêm tài liệu. Vì vậy bạn cần nạp dữ liệu trước khi hỏi đáp.

Luồng ingest hiện có sẵn nằm trong notebook:

- `src/load_datasets.ipynb`: tải dataset từ Hugging Face, chunk văn bản và đẩy vào vector store.

Notebook hiện tại có ví dụ load một phần dataset với `start=15000` và `limit=5000`. Bạn có thể chỉnh lại để phù hợp với dung lượng máy và nhu cầu thử nghiệm.

Dataset nguồn:

- `vohuutridung/vietnamese-legal-documents`

### 5. Chạy ứng dụng

```powershell
python src/main.py
```

Sau đó mở:

- `http://127.0.0.1:8000/`: trang HTML chính.
- `http://127.0.0.1:8000/gradio`: giao diện Gradio độc lập.

Lưu ý: script hiện tại không bật auto-reload. Sau khi sửa code, bạn cần dừng và chạy lại tiến trình server.

## Chạy bằng Docker

### Build image

```bash
docker build -t corporation-law-vietnam-rag .
```

### Chạy container app

```bash
docker run --rm -p 8000:8000 --env-file .env corporation-law-vietnam-rag
```

Nếu app container cần kết nối tới Qdrant đang chạy trên host, trên Windows hoặc macOS bạn thường sẽ cần đặt `QDRANT_HOST=host.docker.internal` trong `.env`. Nếu chạy app và Qdrant trong cùng một Docker network, đặt `QDRANT_HOST` bằng tên service hoặc tên container của Qdrant.

## Cấu trúc thư mục

```text
src/
	api/           FastAPI, HTML template, static files, Gradio mount
	dataset/       Load dataset, chunk document, vector store, SQLite helper
	rag/           Agent, chatbot, embeddings, rerank, model wrappers
	settings/      Cấu hình hệ thống và cache
datasets/        Dataset cache
db/              SQLite và vector store dữ liệu
models/          Model cache và sentence cache
stored/          Thư mục lưu trữ bổ sung
```

## Biến môi trường quan trọng

- `DEEPSEEK_API_KEY`: API key của DeepSeek cho cấu hình mặc định.
- `APP_HOST`: host để Uvicorn bind, mặc định `127.0.0.1`.
- `APP_PORT`: cổng của ứng dụng, mặc định `8000`.
- `QDRANT_HOST`: host của Qdrant, mặc định `localhost`.
- `QDRANT_PORT`: cổng của Qdrant, mặc định `6333`.
- `QDRANT_TIMEOUT`: timeout khi giao tiếp Qdrant.
- `QDRANT_BATCH_SIZE`: kích thước batch khi upsert document.
- `QDRANT_UPSERT_WAIT`: có cho indexing xong mới trả về hay không.

## Luồng xử lý câu hỏi

1. Người dùng gửi câu hỏi qua Gradio.
2. LangGraph phân loại câu hỏi trong `llm_node`.
3. Nếu là câu hỏi pháp lý, hệ thống truy xuất tài liệu từ Qdrant.
4. Kết quả truy xuất được rerank.
5. LLM sinh câu trả lời và bổ sung danh sách tài liệu tham khảo nếu có.

## Ghi chú vận hành

- Lần chạy đầu tiên có thể chậm do cần tải dataset và model.
- Cache được đặt trong chính repo, vì vậy kích thước thư mục có thể tăng nhanh.
- Dockerfile đã tối ưu cho Linux container bằng cách cài CPU-only PyTorch và chuẩn hóa `requirements.txt` trước khi cài.
- Nếu collection Qdrant rỗng, app vẫn chạy nhưng khả năng truy xuất sẽ không đúng như mong đợi.

## Hướng phát triển tiếp

- Bổ sung script ingest dữ liệu bằng `.py` để không phụ thuộc notebook.
- Tách cấu hình provider/model ra file `.env` hoặc settings tập trung.
- Bổ sung auto-reload cho môi trường phát triển.
- Thêm test cho luồng retrieval, rerank và response generation.
# Corporation Law Vietnam RAG

Ứng dụng hỏi đáp về Luật Doanh nghiệp Việt Nam theo mô hình RAG, kết hợp FastAPI, Gradio, LangGraph, Qdrant và mô hình embedding đa ngôn ngữ. Dự án tập trung vào việc truy xuất văn bản pháp luật tiếng Việt, xếp hạng lại ngữ cảnh liên quan, rồi sinh câu trả lời ngắn gọn kèm tài liệu tham khảo khi có.

## Tính năng chính

- Giao diện web đơn giản tại `/` và giao diện chat Gradio độc lập tại `/gradio`.
- Phân luồng câu hỏi bằng LangGraph: câu hỏi pháp lý đi theo nhánh RAG, câu hỏi chung/chit-chat đi theo nhánh trả lời trực tiếp.
- Truy xuất tài liệu từ Qdrant vector store.
- Rerank tài liệu bằng cross-encoder trước khi sinh câu trả lời.
- Tự tạo cache cục bộ cho dataset và model trong thư mục dự án.

## Kiến trúc tổng quan

1. FastAPI render trang HTML và mount Gradio ChatInterface.
2. Người dùng gửi câu hỏi qua Gradio.
3. LangGraph agent phân loại câu hỏi trong `llm_node`.
4. Nếu là câu hỏi pháp lý:
	 - truy xuất ngữ cảnh từ Qdrant,
	 - rerank tài liệu,
	 - sinh câu trả lời từ LLM,
	 - thêm tài liệu tham khảo nếu metadata có URL.
5. Nếu là câu hỏi chung hoặc chit-chat, hệ thống trả lời trực tiếp bằng LLM.

## Công nghệ sử dụng

- Python 3.12
- FastAPI + Jinja2
- Gradio 6
- LangChain + LangGraph
- Qdrant
- Hugging Face Datasets
- sentence-transformers / cross-encoder
- DeepSeek API theo cấu hình hiện tại của mã nguồn

## Cấu trúc thư mục chính

```text
src/
	api/              FastAPI app, template HTML, static assets
	dataset/          Load dataset, chia chunk, vector store, SQLite helpers
	rag/              Agent, nodes, grading, embeddings, LLM wrappers
	settings/         Cấu hình môi trường, cache, hằng số hệ thống
	main.py           Entrypoint chạy Uvicorn
datasets/           Cache dataset Hugging Face
db/                 SQLite và vector store data
models/             Cache model và sentence model
```

## Yêu cầu trước khi chạy

- Windows PowerShell hoặc môi trường Python 3.12 tương đương.
- Qdrant đang chạy và có thể truy cập qua `QDRANT_HOST:QDRANT_PORT`.
- API key cho LLM. Với cấu hình hiện tại, bạn cần `DEEPSEEK_API_KEY`.
- Dữ liệu pháp luật đã được nạp vào Qdrant. Repo này chỉ `build()` collection khi khởi động, không tự ingest toàn bộ dữ liệu.

## Cài đặt local

Các lệnh dưới đây được viết theo Windows PowerShell.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu bạn dùng bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Cấu hình môi trường

Tạo file `.env` ở thư mục gốc dự án:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key

APP_HOST=127.0.0.1
APP_PORT=8000

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_TIMEOUT=120
QDRANT_BATCH_SIZE=16
QDRANT_UPSERT_WAIT=false
```

Biến có sẵn trong code nhưng không dùng mặc định:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`

## Chạy Qdrant

Qdrant phải sẵn sàng trước khi bạn khởi động ứng dụng, vì vector store được build ngay từ lúc import module dataset.

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

## Nạp dữ liệu vào vector store

Nguồn dữ liệu chính trong repo là Hugging Face dataset `vohuutridung/vietnamese-legal-documents`.

Notebook ingest hiện có tại `src/load_datasets.ipynb`. Notebook này thực hiện các bước:

1. Tải dataset content và metadata.
2. Gộp dữ liệu theo `id`.
3. Chia tài liệu thành các chunk phù hợp với văn bản pháp luật.
4. Khởi tạo `VectorStoreDB`.
5. Gọi `db.add(...)` để đẩy dữ liệu vào Qdrant.

Lưu ý:

- `documentDB.build()` trong ứng dụng chỉ đảm bảo collection tồn tại, không tự thêm tài liệu.
- Nếu Qdrant chưa có dữ liệu, ứng dụng vẫn khởi động được nhưng chất lượng trả lời RAG sẽ không đúng mục tiêu.
- Trong notebook, bạn có thể chỉnh `start` và `limit` để ingest theo từng lô thay vì nạp toàn bộ một lần.

## Chạy ứng dụng

```powershell
python src/main.py
```

Sau khi chạy, truy cập:

- Trang chủ: `http://127.0.0.1:8000/`
- Gradio trực tiếp: `http://127.0.0.1:8000/gradio`

## Docker

Build image:

```bash
docker build -t corporation-law-vietnam-rag .
```

Chạy container ứng dụng:

```bash
docker run --rm -p 8000:8000 --env-file .env corporation-law-vietnam-rag
```

Ghi chú cho Docker:

- Dockerfile hiện dùng Python 3.12 slim.
- Dockerfile tự cài CPU-only PyTorch.
- Dockerfile có xử lý trường hợp `requirements.txt` đang ở UTF-16 do được xuất từ PowerShell.
- Nếu ứng dụng cũng chạy trong Docker thì `QDRANT_HOST=localhost` sẽ không trỏ tới máy host. Khi đó hãy dùng `host.docker.internal` hoặc đặt app và Qdrant cùng một Docker network.

## Cache và dữ liệu sinh ra

Ứng dụng tạo và sử dụng các thư mục sau trong repo:

- `datasets/dataset_cache`
- `models/model_cache`
- `models/sentence_cache`
- `db/sql`
- `db/vectorstore`

Các cache này giúp tránh tải lại model và dataset sau mỗi lần chạy.

## Luồng trả lời hiện tại

- Câu hỏi pháp lý: retrieve -> rerank -> answer + reference.
- Câu hỏi kiến thức chung hoặc trò chuyện: trả lời trực tiếp bằng LLM.
- Session chat được gắn theo `request.session_hash` của Gradio.

## Một số lưu ý vận hành

- Mặc định mã nguồn đang khởi tạo provider DeepSeek trong phần LLM wrapper, nên `DEEPSEEK_API_KEY` là biến quan trọng nhất để chạy ngay.
- Nếu bạn muốn đổi sang OpenAI, Google hoặc local model, hãy chỉnh lại phần provider/model trong `src/rag/llm/`.
- Nếu gặp lỗi truy xuất dữ liệu, kiểm tra lại trạng thái Qdrant trước tiên.
- Nếu giao diện mở được nhưng câu trả lời không có ngữ cảnh pháp lý, nguyên nhân thường là collection Qdrant chưa được ingest dữ liệu.