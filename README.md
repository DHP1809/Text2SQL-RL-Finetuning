# Text2SQL Reinforcement Learning Fine-tuning Framework

A specialized system designed for a Two-Stage Training Flow, combining Supervised Fine-Tuning (SFT) and Reinforcement Learning (GRPO) to optimize large language models for the Text-to-SQL task.

---

## 🗂️ Directory Structure and Purposes (Repository Structure)

### 1. `Finetuning/` (Training Source Code)
Centralizes all source code, configurations, and training scripts for both model fine-tuning stages:
* **LLamaFactory (SFT):** Implements supervised fine-tuning based on `Plan-Guided Trajectories`, helping the foundation model familiarize itself with generating complex reasoning code prior to policy optimization.
* **EasyR1 (GRPO):** A dedicated library for implementing the **GRPO (Group Relative Policy Optimization)** reinforcement learning algorithm. It optimizes the model through a centralized Reward System without the need for a cumbersome Critic model configuration, thereby maximizing VRAM savings.

### 2. `arctic_text2sql_r1/` (Evaluation Source Code)
* A specialized directory containing independent evaluation logic (Evaluation Metrics) strictly following the methodology of the **arctic_text2sql** framework.
* Contains the `auto_evaluation.py` script, which is responsible for testing the output SQL queries directly on a Database Sandbox to verify Execution-based Accuracy, rather than mechanically relying on Exact Match string comparison.

### 3. `data_process_pipeline/` (Data Processing Pipeline)
The input data preprocessing pipeline strictly adheres to the design philosophy and data filtering workflow to help the model deeply comprehend the database structure (Database Schema Understanding):
* **Data Filtering & Two-stage Execution Validation:** Executes two-step testing on every SQL query in the original dataset, automatically detecting and completely removing erroneous, empty, or sub-optimal queries to ensure the cleanliness of the training dataset.

---

## 🏋️ Overall Operation Workflow

1. **`data_process_pipeline`** cleans, filters errors, and packages the data based on actual execution feedback from the Database Engine.
2. The processed standard data is loaded into **`Finetuning/`** to run through **LLamaFactory (SFT)** and is subsequently aligned using **EasyR1 (GRPO)**.
3. The post-training model checkpoint is transferred to **`arctic_text2sql_r1/`** for a comprehensive evaluation of metrics and Execution Success Rate before being packaged for the production environment.
