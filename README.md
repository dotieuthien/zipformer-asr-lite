# Zipformer ASR lite

Nhận dạng tiếng nói offline bằng [Zipformer](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer)
qua [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) trên GPU. Câu hỏi: một RTX 3060 phục vụ được bao nhiêu người cùng
lúc, với độ trễ bao nhiêu.

Repo đi theo cách làm của [vieneu](../vieneu): hiểu và đo bản gốc trước, rồi mới làm bản lite. Mọi phép đo chạy trên GPU
thuê theo giờ qua [VKong](https://vkong.tli-tech.com), do AI agent điều khiển từ đầu đến cuối.

## Câu chuyện

**Giai đoạn 1: hiểu server gốc và đo baseline (17/10/2026).**

- Đọc code server offline của sherpa-onnx: nhận audio, gom tối đa 5 câu một batch, chạy encoder rồi greedy search trên
  GPU, trả cả batch cùng lúc.
- Viết thêm một bản Python dịch đúng logic server C++ đó, rồi đo cả hai bằng câu tiếng Anh thật (LibriSpeech).
- Kết quả: bản Python chậm hơn khi tải cao, nên baseline là server C++ gốc.

Chi tiết: [PHASE1.md](PHASE1.md).

**Giai đoạn 2: bản lite (18/10/2026).**

- Viết lại server bằng Python theo cách của vieneu lite: slot cố định, prefill một lần cho cả nhóm câu mới vào, vòng greedy
  chạy trên GPU bằng CUDA graph, câu nào xong là trả ngay.
- Audio dài được cắt thành cửa sổ 30 s rồi ghép lại, theo cách của icefall.
- Kết quả: phục vụ được gấp 2.5–3 lần số người trong mốc 1 s, chữ giống hệt sherpa. Còn thua sherpa khi request rải đều và
  tải cực cao; VRAM đã có trần nhưng chưa phẳng hoàn toàn.

Chi tiết: [OPTIMIZATION.md](OPTIMIZATION.md).

## Kết quả

Server C++ gốc, cấu hình mặc định, model Zipformer gốc tiếng Anh, một RTX 3060 12 GB:

| Kịch bản | Người dùng tối đa, latency P90 ≤ 1 s | Latency P50 / P90 ở mức đó | RTF P50 / P90 ở mức đó | Trần xử lý |
|---|---:|---:|---:|---:|
| Gửi cùng lúc, câu ~10 s | 24 | 0.71 / 0.99 s | 0.072 / 0.099 | 422 s audio/giây |
| Gửi cùng lúc, câu 1.4–33 s | 16 | 0.46 / 0.77 s | 0.083 / 0.124 | 282 s audio/giây |
| Nói liên tục, câu ~10 s | 128 | 0.33 / 0.77 s | 0.034 / 0.077 | ~515 s audio/giây |
| Nói liên tục, câu 1.4–33 s | 64 | 0.20 / 0.33 s | 0.034 / 0.080 | ~112 s audio/giây |

- **Gửi cùng lúc:** N người gửi một câu đúng cùng một lúc. **Nói liên tục:** N người nói không nghỉ, câu nào xong gửi ngay.
- **Latency:** thời gian tới khi có chữ. ASR offline trả cả câu một lần, nên đây cũng là thời gian tới chữ đầu tiên.
- **RTF:** latency chia độ dài câu.
- **Một câu chạy một mình:** latency 104 ms với câu dưới 5 s, 222 ms với câu 20–33 s.
- **Độ chính xác:** WER ~2.2%, không đổi khi tải nặng.
- **Câu dài ngắn lẫn lộn** làm sức chứa giảm khoảng 4 lần.

Bảng theo từng mức tải, so sánh với bản Python, và vì sao sức chứa dừng ở đó: [PHASE1.md](PHASE1.md).

## Chạy thử

Cần CLI của VKong đã đăng nhập:

```bash
vkong run -C baseline --detach       # thuê RTX 3060 12 GB, cài thư viện, tải model và dữ liệu
vkong instances                      # lấy id của máy, dạng vk_xxxx
vkong attach vk_xxxx -C baseline     # in ra http://127.0.0.1:PORT
```

Ở cửa sổ khác:

```bash
URL=http://127.0.0.1:PORT
curl -X POST "$URL/server?impl=cpp"                                                  # bật server C++ gốc ở port 6006
curl -X POST "$URL/benchmark?tag=t1&mode=burst&concurrency=1,8,32&audio=same"        # đo ngay trên máy GPU
curl "$URL/benchmark"                                                                # hỏi đã xong chưa
curl "$URL/results/cpp-b5-w3-t2/t1.log"                                              # xem kết quả
vkong app stop zipformer-asr-baseline --yes                                          # trả máy
```

`Ctrl+C` chỉ ngắt kết nối, máy vẫn tính tiền tới khi `stop`. Cách đo lại cả bộ: [PHASE1.md](PHASE1.md#tự-kiểm-tra-và-chạy-lại).

## Trong repo

| Đường dẫn | Nội dung |
|---|---|
| [baseline/server_cpp.sh](baseline/server_cpp.sh) | Chạy binary C++ gốc của sherpa-onnx, không sửa gì |
| [baseline/server_python.py](baseline/server_python.py) | Bản Python dịch từ server C++, cùng giao thức và flag |
| [baseline/loadtest.py](baseline/loadtest.py) | Client đo: gửi cùng lúc, nói liên tục, latency, RTF, WER |
| [baseline/bench.py](baseline/bench.py) | Chạy trên máy GPU: bật server cần đo, chạy client, trả file kết quả |
| [baseline/model.py](baseline/model.py), [baseline/audio.py](baseline/audio.py) | Tải model, binary C++, LibriSpeech; chọn bộ câu |
| [baseline/results/gpu-20261017/](baseline/results/gpu-20261017/) | Số đo thô của lần đo trong PHASE1.md |
| [experiments/baseline/](experiments/baseline/) | Chạy cả bộ, tải kết quả, tính bảng |
| [lite/](lite/) | Bản lite: [serve.py](lite/serve.py), [zipformer_lite/](lite/zipformer_lite/), cùng bộ đo; số đo thô trong [lite/results/gpu-20261018/](lite/results/gpu-20261018/) |
| [experiments/lite/](experiments/lite/) | Chạy cả bộ cho lite, giữ tunnel, tải kết quả, tính bảng so với sherpa |
