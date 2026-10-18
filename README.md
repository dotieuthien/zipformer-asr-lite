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

**Giai đoạn 2: bản lite và tối ưu (18/10/2026).**

- Viết lại server bằng Python theo cách của vieneu lite: slot cố định, prefill một lần cho cả nhóm câu mới vào, vòng greedy
  chạy trên GPU bằng CUDA graph, câu nào xong là trả ngay.
- Đo lần đầu thì thấy lite thua sherpa khi request đến rải đều và tải cao: encoder và greedy phải chờ nhau trên cùng một
  luồng. Tách encoder ra luồng riêng thì hết.
- Audio dài được cắt thành cửa sổ 30 s rồi ghép lại, theo cách của icefall. VRAM có trần, và tự nhả khi vượt ngưỡng.

Chi tiết: [OPTIMIZATION.md](OPTIMIZATION.md).

## Kết quả

Bản lite so với server C++ gốc của sherpa, model Zipformer gốc tiếng Anh, một RTX 3060 12 GB:

| Kịch bản | Người dùng tối đa (P90 ≤ 1 s), sherpa | Người dùng tối đa (P90 ≤ 1 s), lite | Trần xử lý, sherpa | Trần xử lý, lite |
|---|---:|---:|---:|---:|
| Dồn một lúc, câu ~10 s | 24 | **64** | 422 | **825** |
| Dồn một lúc, câu 1.4–33 s | 16 | **64** | 282 | **552** |
| Rải đều, câu ~10 s | 128 | **768** | 515 | **916** |
| Rải đều, câu 1.4–33 s | 64 | **320** | 112 | **394** |

- **Dồn một lúc:** N người gửi một câu đúng cùng một lúc. **Rải đều:** N người nói không nghỉ, câu nào xong gửi ngay, nên
  request đến rải rác, khoảng N chia độ dài câu mỗi giây.
- **Thời gian chờ:** tới khi có chữ. ASR offline trả cả câu một lần, nên đây cũng là thời gian tới chữ đầu tiên.
- **Trần xử lý:** số giây audio chép ra chữ trong một giây khi server đã hết sức.
- **Một câu ~10 s chạy một mình:** sherpa 0.135 s, lite 0.076 s.
- **Chữ không đổi:** 355/355 câu của lite giống hệt sherpa; WER ~2.2%, không đổi khi tải nặng.

Server gốc của sherpa hoạt động thế nào và vì sao sức chứa dừng ở đó: [PHASE1.md](PHASE1.md). Lite làm gì khác, từng
bước tối ưu và cách đo: [OPTIMIZATION.md](OPTIMIZATION.md).

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

Bản lite: thay `baseline` bằng `lite`, `/server?impl=cpp` bằng `/server?name=lite`, và tên app thành
`zipformer-asr-lite`. Sửa code xong thì `vkong attach vk_xxxx -C lite --update`, không cần thuê lại.

`Ctrl+C` chỉ ngắt kết nối, máy vẫn tính tiền tới khi `stop`. Cách đo lại cả bộ: [PHASE1.md](PHASE1.md#tự-kiểm-tra-và-chạy-lại)
cho sherpa, [OPTIMIZATION.md](OPTIMIZATION.md#bước-8-bench-cuối-so-với-sherpa) cho lite.

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
| [lite/serve.py](lite/serve.py) | Server websocket của bản lite, cùng giao thức với sherpa |
| [lite/zipformer_lite/](lite/zipformer_lite/) | Slot, luồng prefill và luồng greedy, cửa sổ cho audio dài, nhả VRAM |
| [lite/results/](lite/results/) | Số đo thô của giai đoạn 2: `gpu-20261018` (bước 3–5), `gpu-20261018b` (bước 6–8) |
| [experiments/lite/](experiments/lite/) | Chạy cả bộ cho lite, giữ tunnel, tải kết quả, tính bảng so với sherpa |
