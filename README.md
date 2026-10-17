# Zipformer ASR lite

Nhận dạng tiếng nói offline bằng [Zipformer](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer)
chạy qua [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) trên GPU. Câu hỏi của giai đoạn này: một RTX 3060 phục
vụ được bao nhiêu người dùng cùng lúc, khi mọi câu dài bằng nhau và khi câu dài ngắn lẫn lộn.

Repo đi theo cách làm của [vieneu](../vieneu): đo bản gốc (baseline) trước, sau đó mới làm bản lite. Mọi phép đo chạy
trên GPU thuê theo giờ qua [VKong](https://vkong.tli-tech.com), do AI agent điều khiển từ đầu đến cuối.

**Giai đoạn 1: hiểu server gốc.** Server offline của sherpa-onnx gồm những phần nào, một request đi vào và đi ra ra sao,
kỹ thuật bên trong, và vì sao sức chịu tải dừng ở mức đo được: [PHASE1.md](PHASE1.md). README này là số đo của server
gốc, kèm phép so sánh với bản port Python.

## Kết quả

Server C++ gốc của sherpa-onnx, cấu hình mặc định, model Zipformer gốc tiếng Anh, một RTX 3060 12 GB (17/10/2026):

| Kịch bản | Người dùng tối đa, P90 ≤ 1 s | Người dùng tối đa, P90 ≤ 2 s | Quá tải ở | Trần xử lý |
|---|---:|---:|---:|---:|
| Gửi cùng lúc, câu ~10 s | **24** | 64 | — | 422 s audio/giây (ở 128 người) |
| Gửi cùng lúc, câu 1.4–33 s | **16** | 64 | — | 282 s audio/giây (ở 128 người) |
| Nói liên tục, câu ~10 s | **128** | 512 | 640 người | ~515 s audio/giây |
| Nói liên tục, câu 1.4–33 s | **64** | 64 | 128 người | ~112 s audio/giây |

- **Gửi cùng lúc:** N người bấm gửi đúng cùng một lúc, mỗi người một câu. Đây là lúc dồn tải tệ nhất.
- **Nói liên tục:** N người nói không nghỉ. Mỗi câu được gửi ngay khi nói xong, và mỗi người chỉ có một câu đang chờ
  kết quả. Kịch bản này gần với một tổng đài hay một phòng họp có nhiều người nói cùng lúc.
- **P90:** 90% request nhận được chữ trong khoảng thời gian này, tính từ lúc gửi (gửi cùng lúc) hoặc từ lúc câu nói xong
  (nói liên tục, gồm cả thời gian xếp hàng).
- **Quá tải:** server không xử lý kịp lượng audio được gửi tới, hàng đợi dài dần và thời gian chờ tăng không dừng.
- **Trần xử lý:** số giây audio server chép ra chữ trong một giây khi đã bão hòa.

Rút ra:

- **Câu dài ngắn lẫn lộn làm sức chứa giảm khoảng 4 lần.** Khi nói liên tục, server chịu được ~515 giây audio mỗi giây
  nếu mọi câu dài ~10 s, nhưng chỉ ~112 giây audio mỗi giây với câu dài tự nhiên 1.4–33 s. Một phần là do đệm: câu
  ngắn chung batch với câu dài nên GPU phải tính gần gấp đôi lượng audio thật (xem
  [phần dưới](#khi-đông-người-kết-quả-trả-về-thế-nào)).
- **GPU chưa phải nút thắt.** Ngay cả khi quá tải, GPU chỉ bận 24–63% và server dùng 2–2.8 nhân CPU, còn mỗi batch đã
  gần chạm mức tối đa 5 câu (trung bình 4.6–5.0). Chỗ cần xem cho bản lite là cách gom batch và phần chạy trên CPU,
  không phải tốc độ tính của GPU. Đây là giả thuyết, chưa đo.
- **Không sai chữ khi tải nặng.** WER lúc tải bằng WER lúc chạy riêng: 1.95% với 55 câu ~10 s, 2.20% với 300 câu lẫn
  lộn. Con số này khớp với số icefall công bố cho đúng checkpoint này, 2.22–2.27% với greedy search trên toàn bộ
  test-clean. Tổng cộng 34,666 request, không lỗi nào.

## Khi đông người, kết quả trả về thế nào

Kết quả không trả về cùng một lúc cho cả N người, cũng không phải câu nào xong thì trả câu đó. **Kết quả trả về theo
từng batch:**

- Server gom tối đa 5 câu thành một batch và giải mã cả batch trên GPU.
- Khi batch xong, cả 5 kết quả được gửi đi cùng lúc. Trong `offline-websocket-server-impl.cc`, hàm `Decode()` gọi
  `DecodeStreams` cho cả batch rồi mới gửi từng kết quả.
- Ba luồng làm việc chạy ba batch song song. Batch nào xong trước thì 5 người của batch đó nhận kết quả trước.

Số đo cho thấy đúng như vậy. Ở mức 128 người gửi cùng lúc (5 lượt, 640 request), server C++ trả kết quả thành
khoảng 140 nhóm. Trong đó 123–124 nhóm có đúng 5 request nhận kết quả cách nhau dưới 2 ms. Diễn biến một lượt 128 người gửi cùng lúc, câu 1.4–33 s:

| Đợt | Trả về lúc | Số request | Độ dài các câu |
|---:|---:|---:|---|
| 1 | 0.21 s | 1 | 2.1 s |
| 2 | 0.22 s | 1 | 2.2 s |
| 3 | 0.23 s | 1 | 1.9 s |
| 4 | 0.41 s | 5 | 2.0, 2.0, 2.4, 2.6, 2.7 s |
| 5 | 0.49 s | 5 | 1.7, 2.9, 3.2, 3.4, 3.9 s |
| … | | | |
| 26 | 3.13 s | 5 | 12.4, 13.0, 13.3, 13.8, 14.2 s |
| 27 | 3.42 s | 5 | 14.4, 15.9, 16.7, 18.0, 18.6 s |
| 28 | 3.54 s | 5 | 18.1, 19.1, 19.9, 22.6, 28.6 s |

128 người nhận kết quả rải ra trong 28 đợt, từ 0.21 s tới 3.54 s, trung bình cách nhau 0.12 s. Ba đợt đầu chỉ có 1 câu,
vì ba luồng làm việc đang rảnh nên mỗi luồng lấy ngay câu vừa tới. Từ đợt 4 trở đi hàng đợi đã dồn lên, nên mỗi đợt
đủ 5 câu.

**Cái giá của việc trả theo batch:** câu ngắn phải chờ câu dài nhất trong cùng batch. GPU tính cả batch theo độ dài câu
dài nhất, phần còn lại của các câu ngắn được đệm cho bằng. Mức đệm khác nhau giữa hai kịch bản:

| Kịch bản (128 người, C++ gốc) | Batch trông thế nào | Audio GPU phải tính / audio thật |
|---|---|---:|
| Gửi cùng lúc, câu ~10 s | Mọi câu dài như nhau | 1.01× |
| Gửi cùng lúc, câu 1.4–33 s | Câu ngắn tải lên xong trước nên vào hàng trước. Batch tự xếp theo độ dài, như bảng trên | 1.17× |
| Nói liên tục, câu 1.4–33 s | Câu đến ngẫu nhiên, một batch lẫn cả câu ngắn lẫn câu dài, ví dụ 1.7, 2.0, 3.2, 4.3 và 14.4 s | 1.86× |

Khi nói liên tục với câu lẫn lộn, GPU phải tính gần gấp đôi lượng audio thật. Đây là một phần lý do sức chứa giảm
khoảng 4 lần so với câu ~10 s. Phần còn lại chưa đo: câu ngắn hơn nên có nhiều request hơn cho cùng một lượng audio,
và greedy search chạy theo số khung của câu dài nhất. Bản Python cho kết quả tương tự (1.80×).

Cách tính: các request nhận kết quả cách nhau dưới 2 ms được xem là cùng một batch. Thỉnh thoảng hai batch xong trùng
nhau nên có nhóm 8–10 request. Tính lại bằng `python3 experiments/baseline/analyze.py`, mục "Kết quả trả về theo batch".

## C++ gốc so với bản port Python

Baseline phải là code gốc của sherpa. Mình vẫn viết thêm một bản Python dịch đúng logic của server C++, vì Python dễ sửa
để làm bản lite. Điều kiện: bản Python phải chạy ngang bản C++ khi đo cùng một cách.

| Kịch bản | P90 ≤ 1 s, C++ | P90 ≤ 1 s, Python | Trần xử lý, C++ | Trần xử lý, Python |
|---|---:|---:|---:|---:|
| Gửi cùng lúc, câu ~10 s | 24 | 8 | 422 s/giây | 304 s/giây |
| Gửi cùng lúc, câu 1.4–33 s | 16 | 24 | 282 s/giây | 260 s/giây |
| Nói liên tục, câu ~10 s | 128 | 128 | ~515 s/giây | ~360 s/giây |
| Nói liên tục, câu 1.4–33 s | 64 | 64 | ~112 s/giây | ~124–132 s/giây |

**Kết luận: bản Python chưa ngang C++, nên baseline là server C++ gốc.**

- **Tải thấp** (tới 8 người gửi cùng lúc, tới 128 người nói liên tục): hai bản như nhau.
- **Gửi cùng lúc, câu ~10 s:** từ 16 người trở lên, bản Python chậm hơn đều ở mọi mức. P90 là 1.04 s so với 0.87 s ở 16
  người, và 1.28 s so với 0.99 s ở 24 người. Vì vậy mốc 1 s của bản Python rơi ở 8 người thay vì 24.
- **Gửi cùng lúc, câu lẫn lộn:** bản Python nhỉnh hơn (24 so với 16 người). Nhưng ở 24–32 người, P90 của cả hai đều
  quanh 0.9–1.2 s, nên coi như ngang nhau.
- **Tải cao với câu ~10 s:** bản Python chỉ đạt khoảng 70% trần xử lý của C++. Khi nói liên tục, bản Python quá tải từ
  384 người, trong khi C++ chịu được tới 512.
- **Câu lẫn lộn:** cả hai cùng bị giới hạn quanh 110–130 giây audio mỗi giây. Ở mức 128 người, C++ sập còn Python vẫn
  trụ được (P90 1.6 s). Tải đang sát trần nên chỉ cần lệch nhỏ cũng ra kết quả khác nhau. Giữa 64 và 128 người chưa đo.
- **Vì sao Python chậm hơn khi tải cao:** chưa đo. Model và engine bên dưới là một, nên phần chậm nằm ở lớp Python. Ứng
  viên là vòng asyncio duy nhất phải nhận từng mẩu 10 KB (một câu 10 s là 63 mẩu), và đoạn giữ GIL khi đổi mảng numpy
  sang C++.

## Cách đo

### Model

[sherpa-onnx-zipformer-en-2023-06-26](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html):

- Checkpoint gốc của Zipformer trong icefall (`Zengwei/icefall-asr-librispeech-zipformer-2023-05-15`).
- Bản medium khoảng 65M tham số, train trên LibriSpeech.
- Tác giả sherpa export thẳng ra ONNX, không train lại. Dùng bản fp32, giải mã greedy search.

### Hai server

| | C++ gốc | Python port |
|---|---|---|
| Code | `sherpa-onnx-offline-websocket-server`, binary build sẵn của release v1.13.8, bản CUDA 12 + cuDNN 9, chạy nguyên | [server_python.py](baseline/server_python.py), dịch từ `offline-websocket-server-impl.cc` của cùng phiên bản |
| Chạy bằng | [server_cpp.sh](baseline/server_cpp.sh) | `python server_python.py` |
| Engine | onnxruntime 1.28.2, CUDA | onnxruntime 1.28.2, CUDA (wheel `sherpa-onnx==1.13.8+cuda12.cudnn9`) |
| Cấu hình | mặc định: `--max-batch-size=5 --num-work-threads=3 --num-threads=2`, `--provider=cuda` | cùng flag |

Cả hai làm giống nhau:

1. Nhận audio qua websocket.
2. Đưa mọi câu vào một hàng đợi chung.
3. Ba luồng làm việc, mỗi luồng lấy tối đa 5 câu, tính đặc trưng rồi giải mã cả batch trên GPU.

Kết quả thô nằm trong thư mục đặt theo cấu hình: `cpp-b5-w3-t2` là C++, batch 5, 3 luồng làm việc, 2 luồng onnxruntime.

### Audio

LibriSpeech test-clean, tức câu tiếng Anh đọc thật, bộ test chuẩn của chính model này. Mỗi câu có transcript gốc. Tải từ
bản sao trên HuggingFace (`openslr/librispeech_asr`); [audio.py](baseline/audio.py) chọn hai bộ:

| Bộ | Câu | Độ dài | Dùng cho |
|---|---:|---|---|
| `same` | 55, tất cả các câu dài 9.75–10.25 s | trung bình 10.00 s, tổng 9.2 phút | câu ~10 s |
| `mixed` | 300, lấy ngẫu nhiên (seed 0) trong 2620 câu | 1.42–32.77 s, trung bình 7.23 s, tổng 36.2 phút | câu 1.4–33 s |

Người dùng thứ `u` gửi câu thứ `(u + k·N) mod n` ở lượt thứ `k`, nên những người gửi cùng lúc gửi câu khác nhau. Nghe
thử một bộ câu: `cd baseline && python audio.py wavs mixed /tmp/mixed`.

### Client đo

[loadtest.py](baseline/loadtest.py) là phần tự viết duy nhất trên đường đo. Cả hai server dùng chung client này:

- **Nói đúng giao thức của client mẫu sherpa** (`offline-websocket-client-decode-files-paralell.py`):
  - gửi 4 byte sample rate, 4 byte độ dài, rồi các sample float32, cắt thành mẩu 10240 byte;
  - nhận một chuỗi JSON kết quả;
  - gửi `Done` khi kết thúc.
- **Chạy ngay trên máy GPU**, như ở vieneu:
  - [bench.py](baseline/bench.py) trên port 8000 bật server cần đo ở port 6006, chạy client và trả file kết quả. Nó không
    nằm trên đường đi của request.
  - Độ trễ mạng giữa máy mình và máy GPU không lọt vào số đo nào.
  - Tunnel VKong có đứt giữa chừng thì phép đo vẫn chạy tiếp.
- **Không để client thành nút thắt:** người dùng được chia cho 8 tiến trình. Mỗi người giữ một kết nối suốt một mức tải,
  mở trước khi bắt đầu tính giờ.

Hai kịch bản, mỗi kịch bản tăng dần số người dùng N:

| | Gửi cùng lúc | Nói liên tục |
|---|---|---|
| Cách gửi | N người gửi một câu cùng lúc, 2 lượt khởi động rồi 5 lượt đo | Mỗi người nói không nghỉ, câu nào nói xong gửi ngay; người dùng bắt đầu lệch nhau |
| Thời gian | Đến khi đủ 5 lượt | 60 s mỗi mức, bỏ 10 s đầu |
| Thước đo thời gian | Từ byte đầu gửi đi đến khi nhận kết quả | Từ lúc câu nói xong đến khi nhận kết quả, gồm cả thời gian xếp hàng |
| Các mức N | 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128 | 32, 64, 128, 192, …, 768 |

- Mức đạt: mọi request thành công và P90 ≤ 1 s.
- Một mức có P90 quá 5 s thì dừng, không đo các mức cao hơn.
- P50/P90 tính trên mọi request của một mức, nội suy tuyến tính như ở vieneu.

Kiểm tra chất lượng:

- Trước mỗi kịch bản, client chép từng câu trong bộ một mình, lần lượt từng câu. Kết quả này là mốc so sánh và cho ra
  WER lúc chạy riêng.
- Lúc có tải, mọi câu trả về được tính WER so với transcript gốc, và so với kết quả lúc chạy riêng.
- Ở các kịch bản câu lẫn lộn, 99–100% câu khớp hoàn toàn với lúc chạy riêng. Vài câu lệch nhẹ do được đệm thêm khi chung
  batch với câu dài hơn, nhưng WER không đổi.

### Máy đo

| Thông số | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12 GB, driver 580.126.20, giới hạn công suất 170 W |
| GPU UUID | `GPU-5cc6a8b4-ff0b-0ab0-21ca-3b4128a5a0e7` |
| CPU | 24 nhân (container `8e87f284f6e3`) |
| Máy thuê | VKong `vk_8b2b7bfcf4da4545`, Hàn Quốc, $0.084/giờ |
| Image | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, Python 3.11.13 |
| Xung nhịp GPU | Không khóa được (máy thuê từ chối `nvidia-smi -lgc`); 1807–1957 MHz trong lúc đo |
| Nhiệt độ GPU | 45–76 °C trong lúc đo |
| Thời gian | 16:33–18:01 ngày 17/10/2026; C++ đo trước, Python đo sau |
| Chi phí | Khoảng 2 giờ 46 phút máy, gồm cả cài đặt và các lần thử, khoảng $0.23 |

### Giới hạn

- **Một máy, một lần đo mỗi mức, đo C++ xong mới tới Python.** Không đo xen kẽ kiểu ABBA như vieneu, nên nếu máy nhanh
  chậm khác nhau theo thời gian thì hai bản không chịu như nhau.
- **Ít request ở mức nhỏ.** Ở kịch bản gửi cùng lúc, mức N có 5·N request. Với N nhỏ, P90 chỉ dựa trên vài chục request.
- **Các mức đo còn thưa.** Kịch bản nói liên tục với câu lẫn lộn chỉ có 64 và 128; sức chứa thật nằm đâu đó giữa hai mức
  này.
- **Mới đo cấu hình mặc định.** Chưa thử đổi `--max-batch-size` hay số luồng.
- **Client và server chung một máy.** Ở các mức tải cao, client dùng dưới 2 nhân trên 24 nhân.

## Số đo chi tiết

Tính lại toàn bộ phần này từ file thô, không cần GPU: `python3 experiments/baseline/analyze.py`.

Mỗi ô là P50 / P90 thời gian chờ, kèm số giây audio xử lý được mỗi giây.

**Gửi cùng lúc, câu ~10 s**

| Người dùng | C++ gốc | Python port |
|---:|---:|---:|
| 1 | 0.13 / 0.15 s · 76 s/giây | 0.14 / 0.14 s · 73 s/giây |
| 8 | 0.53 / 0.75 s · 132 s/giây | 0.48 / 1.00 s · 121 s/giây |
| 16 | 0.68 / 0.87 s · 188 s/giây | 0.71 / 1.04 s · 164 s/giây |
| 24 | 0.71 / 0.99 s · 249 s/giây | 0.77 / 1.28 s · 191 s/giây |
| 32 | 0.96 / 1.33 s · 236 s/giây | 1.26 / 1.72 s · 186 s/giây |
| 64 | 1.11 / 1.65 s · 364 s/giây | 1.59 / 2.22 s · 267 s/giây |
| 128 | 1.81 / 2.75 s · 422 s/giây | 2.55 / 3.78 s · 304 s/giây |

**Gửi cùng lúc, câu 1.4–33 s**

| Người dùng | C++ gốc | Python port |
|---:|---:|---:|
| 1 | 0.13 / 0.19 s · 85 s/giây | 0.13 / 0.21 s · 80 s/giây |
| 8 | 0.49 / 0.62 s · 116 s/giây | 0.29 / 0.59 s · 129 s/giây |
| 16 | 0.46 / 0.77 s · 157 s/giây | 0.46 / 0.68 s · 170 s/giây |
| 24 | 0.56 / 1.16 s · 137 s/giây | 0.53 / 0.92 s · 172 s/giây |
| 32 | 0.57 / 1.22 s · 176 s/giây | 0.68 / 1.11 s · 174 s/giây |
| 64 | 0.89 / 1.98 s · 207 s/giây | 1.07 / 2.10 s · 194 s/giây |
| 128 | 1.51 / 2.81 s · 282 s/giây | 1.45 / 2.85 s · 260 s/giây |

Câu dài phải chờ lâu hơn: ở 64 người, P90 của server C++ là 0.82 s với câu dưới 5 s, nhưng 2.49 s với câu trên 20 s.

**Nói liên tục, câu ~10 s**

| Người dùng | C++ gốc | Python port |
|---:|---:|---:|
| 32 | 0.15 / 0.19 s · 32 s/giây | 0.17 / 0.23 s · 32 s/giây |
| 64 | 0.22 / 0.71 s · 64 s/giây | 0.25 / 0.72 s · 64 s/giây |
| 128 | 0.33 / 0.77 s · 127 s/giây | 0.49 / 0.92 s · 127 s/giây |
| 192 | 0.86 / 1.31 s · 194 s/giây | 0.86 / 1.24 s · 190 s/giây |
| 256 | 0.91 / 1.31 s · 253 s/giây | 1.37 / 2.07 s · 254 s/giây |
| 320 | 0.87 / 1.28 s · 317 s/giây | 2.62 / 3.54 s · 309 s/giây |
| 384 | 0.85 / 1.20 s · 387 s/giây | 6.09 / 8.72 s · 357 s/giây (quá tải) |
| 512 | 1.32 / 1.73 s · 506 s/giây | — |
| 640 | 9.46 / 14.26 s · 515 s/giây (quá tải) | — |

**Nói liên tục, câu 1.4–33 s**

| Người dùng | C++ gốc | Python port |
|---:|---:|---:|
| 32 | 0.16 / 0.25 s · 32 s/giây | 0.18 / 0.26 s · 32 s/giây |
| 64 | 0.20 / 0.33 s · 63 s/giây | 0.21 / 0.46 s · 63 s/giây |
| 128 | 3.56 / 7.03 s · 112 s/giây (quá tải) | 0.95 / 1.59 s · 124 s/giây |
| 192 | — | 12.23 / 27.73 s · 132 s/giây (quá tải) |

Khi nói liên tục, server còn theo kịp thì số giây audio xử lý mỗi giây bằng đúng số người dùng: N người nói không nghỉ
tạo ra N giây audio mỗi giây. Khi con số này thấp hơn N là server đã quá tải.

## Chạy lại

Cần CLI của VKong đã đăng nhập. Chạy từ thư mục gốc của repo.

**Bước 1. Thuê máy.** Lệnh này thuê RTX 3060 12 GB, cài thư viện, tải model, binary C++ và LibriSpeech, rồi chạy `bench.py`:

```bash
vkong run -C baseline --detach
vkong instances                      # lấy id của máy, dạng vk_xxxx
vkong attach vk_xxxx -C baseline     # in ra http://127.0.0.1:PORT; để cửa sổ này mở
```

**Bước 2. Đo.** Ở cửa sổ khác:

```bash
URL=http://127.0.0.1:PORT
bash experiments/baseline/run_suite.sh $URL cpp       # 4 kịch bản trên server C++ gốc, khoảng 1 giờ
bash experiments/baseline/run_suite.sh $URL python    # như trên với bản port Python
```

- Đổi cấu hình bằng ba tham số cuối, ví dụ `run_suite.sh $URL cpp 16 3 2` để đo với batch 16.
- Tunnel đứt thì phép đo vẫn chạy trên máy GPU. Nối lại bằng `vkong attach vk_xxxx -C baseline` (không có `--update`),
  rồi chạy tiếp các kịch bản còn lại, ví dụ `SCENARIOS="sustain-mixed" bash experiments/baseline/run_suite.sh $URL cpp`.

**Bước 3. Tải kết quả, trả máy và tính bảng.** `Ctrl+C` chỉ đóng kết nối, máy vẫn tính tiền tới khi `stop`:

```bash
python3 experiments/baseline/fetch.py $URL --run my-rerun
vkong app stop zipformer-asr-baseline --yes
python3 experiments/baseline/analyze.py --run my-rerun
```

Cũng điều khiển tay được qua HTTP:

| Lệnh | Việc |
|---|---|
| `curl -X POST "$URL/server?impl=cpp&max_batch_size=5&num_work_threads=3&num_threads=2"` | Bật server C++ (hoặc `impl=python`) ở port 6006 |
| `curl -X POST "$URL/benchmark?tag=t1&mode=sustain&concurrency=64,128&audio=mixed"` | Chạy client đo trên máy GPU |
| `curl "$URL/benchmark"` | Hỏi đã đo xong chưa |
| `curl "$URL/results/cpp-b5-w3-t2/t1.log"` | Xem log, kết quả nằm trong `t1/summary.json` |

## Trong repo

| Đường dẫn | Nội dung |
|---|---|
| [baseline/server_cpp.sh](baseline/server_cpp.sh) | Chạy binary C++ gốc của sherpa-onnx, không sửa gì |
| [baseline/server_python.py](baseline/server_python.py) | Bản port Python của server C++, cùng giao thức và flag |
| [baseline/loadtest.py](baseline/loadtest.py) | Client đo: gửi cùng lúc, nói liên tục, WER |
| [baseline/bench.py](baseline/bench.py) | Chạy trên máy GPU: bật server cần đo, chạy client, trả file kết quả |
| [baseline/model.py](baseline/model.py), [baseline/audio.py](baseline/audio.py) | Tải model, binary C++, LibriSpeech; chọn bộ câu |
| [baseline/bench_support.py](baseline/bench_support.py) | Lấy mẫu GPU, CPU mỗi giây |
| [baseline/vkong.yaml](baseline/vkong.yaml) | Cấu hình máy thuê |
| [baseline/results/gpu-20261017/](baseline/results/gpu-20261017/) | Số đo thô: mỗi request một dòng trong `c<N>.jsonl`, tổng hợp trong `summary.json`, log của server |
| [experiments/baseline/](experiments/baseline/) | `run_suite.sh` (chạy cả bộ), `fetch.py` (tải kết quả), `analyze.py` (tính bảng) |
