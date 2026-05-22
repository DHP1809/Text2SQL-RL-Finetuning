# 🚀 Hướng dẫn Huấn luyện LLM: SFT (LLaMA-Factory) & GRPO (EasyR1)

Repository này cung cấp hướng dẫn từng bước (pipeline) để tinh chỉnh các Mô hình Ngôn ngữ Lớn (LLMs). Quy trình được chia làm hai giai đoạn chính nhằm tối ưu hóa khả năng tuân thủ chỉ thị và tư duy suy luận của mô hình:

1. **Phase 1: Supervised Fine-Tuning (SFT)** sử dụng [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
2. **Phase 2: Group Relative Policy Optimization (GRPO)** sử dụng [EasyR1](https://github.com/EasyR1-Team/EasyR1) để tối ưu hóa khả năng suy luận (reasoning) lấy cảm hứng từ DeepSeek-R1.

---

## 📋 Yêu cầu hệ thống (Prerequisites)

- OS: Linux (Ubuntu 20.04/22.04 khuyến nghị)
- GPU: NVIDIA GPU (VRAM >= 24GB cho model 7B/8B, tùy thuộc vào việc dùng LoRA/QoRA)
- CUDA Toolkit: >= 11.8
- Python: >= 3.10

### 🛠 Cài đặt môi trường

```bash
# Tạo conda environment
conda create -n llm-train python=3.10 -y
conda activate llm-train

# Cài đặt PyTorch (thay đổi tùy theo phiên bản CUDA của bạn)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🎯 Phase 1: Supervised Fine-Tuning (SFT) với LLaMA-Factory

Trong giai đoạn này, chúng ta sẽ dạy mô hình cách trả lời theo một định dạng cụ thể (Instruction Tuning) bằng cách sử dụng LLaMA-Factory.

### 1. Cài đặt LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 2. Chuẩn bị dữ liệu

LLaMA-Factory hỗ trợ định dạng chuẩn của Alpaca hoặc ShareGPT. Bạn cần khai báo dataset của mình vào file `data/dataset_info.json`.

Ví dụ một file `sft_data.json` (Định dạng Alpaca):
```json
[
  {
    "instruction": "Bạn là ai?",
    "input": "",
    "output": "Tôi là một trợ lý AI hữu ích."
  }
]
```

### 3. Chạy Huấn luyện SFT (sử dụng LoRA)

Bạn có thể sử dụng WebUI để thiết lập dễ dàng:
```bash
llamafactory-cli webui
```

Hoặc chạy trực tiếp qua CLI bằng file config (ví dụ: `sft_config.yaml`):

```yaml
# sft_config.yaml
model_name_or_path: meta-llama/Llama-3-8B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
dataset: sft_data
template: llama3
output_dir: saves/llama3-8b/sft/lora
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
```

Chạy lệnh train:
```bash
llamafactory-cli train sft_config.yaml
```

### 4. Merge LoRA weights (Tùy chọn)
Sau khi SFT, bạn cần gộp (merge) trọng số LoRA vào model gốc để chuẩn bị cho Phase 2.

```bash
llamafactory-cli export merge_config.yaml
```

---

## 🧠 Phase 2: Reinforcement Learning (GRPO) với EasyR1

Sau khi mô hình đã biết cách trả lời (SFT), chúng ta sử dụng GRPO để tối ưu hóa chuỗi suy luận (Chain of Thought), giúp mô hình giải quyết các bài toán logic hoặc toán học tốt hơn.

### 1. Cài đặt EasyR1

```bash
git clone https://github.com/EasyR1-Team/EasyR1.git
cd EasyR1
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu cho GRPO

GRPO không cần nhãn chi tiết từng bước, mà chỉ cần **Prompt** và **Reward Function** (hàm tính điểm). Dữ liệu thường bao gồm câu hỏi và đáp án đúng để code Python tự động chấm điểm output của mô hình.

Prompt sẽ được tạo trong folder EasyR1/examples/format_prompt, định dạng của file prompt sẽ là `.jinja`.
Định dạng dữ liệu prompt mẫu:
You are playing a number selection game. Your goal is to select the CORRECT number based on the traffic light color.

```bash
{{ content | trim }} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.
``` 

"content" ở đây chính là input nhập vào từ tập dữ liệu training. 

### 3. Cấu hình Reward Functions (Hàm phần thưởng)

EasyR1 cho phép bạn định nghĩa các hàm phần thưởng (ví dụ: thưởng khi trả lời đúng định dạng `<think>...</think><answer>...</answer>`, thưởng khi đáp án toán học chính xác). Bạn cần kiểm tra và chỉnh sửa file định nghĩa reward trong repo, cụ thể là ở đường dẫn EasyR1/examples/reward_function (ví dụ: `rewards.py`).

### 4. Cấu hình file YAML cho EasyR1
EasyR1 quản lý toàn bộ quá trình train (Actor, Critic, Rollout, Reward) thông qua một file cấu hình. Tuỳ chỉnh lại file config.yaml:

```bash
data:
  train_files: ./dataset/training_data.json
  val_files: ./dataset/validate_data.json
  prompt_key: input_seq
  answer_key: output_seq
  image_key: images
  video_key: videos
  ...
```

### 5. Chạy Huấn luyện GRPO

Sử dụng lệnh khởi chạy của EasyR1. Đảm bảo bạn trỏ `model_path` tới thư mục model SFT đã merge ở Phase 1.

```bash
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=8 \
    trainer.experiment_name=qwen2_5_vl_32b_geo_grpo \
    trainer.n_gpus_per_node=8
```
*(Lưu ý: Các tham số script cụ thể có thể thay đổi tùy thuộc vào bản cập nhật mới nhất của EasyR1.)*.

---

## 💡 Lưu ý quan trọng (Best Practices)

- **VRAM Out of Memory (OOM):** Quá trình GRPO thường tốn bộ nhớ hơn SFT rất nhiều do phải sinh text (generation) trong lúc train. Hãy cân nhắc sử dụng Deepspeed Zero-3, kích hoạt `gradient_checkpointing`, hoặc giảm `max_completion_length`.
- **Định dạng Prompt:** GRPO cực kỳ nhạy cảm với format của prompt. Hãy đảm bảo `template` sử dụng trong LLaMA-Factory và template trong EasyR1 đồng nhất với nhau.
- **Theo dõi quá trình:** Luôn sử dụng Weights & Biases (`wandb`) để theo dõi các metrics như loss, reward score và KL divergence trong quá trình chạy GRPO.

---

## 🤝 Đóng góp (Contributing)
Mọi đóng góp nhằm cải thiện pipeline này đều được hoan nghênh! Vui lòng mở Issue hoặc tạo Pull Request.