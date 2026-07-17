# 官方 UrbanGPT 评估与可解释性改造执行方案

本文档用于指导在云服务器上直接评估官方 UrbanGPT checkpoint，并在后续加入 xTP-LLM 风格的可解释性机制。当前目标不是重新训练模型，而是：

1. 下载并评估官方 UrbanGPT 模型。
2. 得到 NYC taxi zero-shot 评估结果。
3. 在现有预测结果基础上增加交通流预测解释。

## 1. 当前本地目录检查结论

当前仓库根目录结构已经基本正确：

```text
UrbanGPT/
├── checkpoints/
├── doc/
├── instruction_generate/
├── metric_calculation/
├── playground/
├── ST_data_urbangpt/
├── tests/
├── urbangpt/
├── requirements.txt
├── urbangpt_eval.sh
└── urbangpt_train.sh
```

### 1.1 已准备完成

评估代码入口存在：

```text
urbangpt/eval/run_urbangpt.py
urbangpt/model/ST_Llama.py
urbangpt/conversation.py
```

ST-Encoder 权重存在：

```text
checkpoints/st_encoder/pretrain_stencoder.pth
```

Vicuna-7B-v1.5-16k 基座模型文件存在：

```text
checkpoints/vicuna-7b-v1.5-16k/config.json
checkpoints/vicuna-7b-v1.5-16k/pytorch_model-00001-of-00002.bin
checkpoints/vicuna-7b-v1.5-16k/pytorch_model-00002-of-00002.bin
checkpoints/vicuna-7b-v1.5-16k/tokenizer.model
```

NYC taxi zero-shot 评估数据已解压：

```text
ST_data_urbangpt/NYC_taxi_cross-region/NYCtaxi_zeroshot.json
ST_data_urbangpt/NYC_taxi_cross-region/NYCtaxi_zeroshot_pkl.pkl
```

训练数据也已解压，但当前任务不需要重新训练：

```text
ST_data_urbangpt/train_data/NYCmulti.json
ST_data_urbangpt/train_data/NYCmulti_pkl.pkl
```

### 1.2 仍需完成

官方 UrbanGPT checkpoint 目录当前为空：

```text
checkpoints/UrbanGPT_pretrained/
```

因此现在还不能直接运行官方模型评估。必须先下载官方 checkpoint。

评估指标脚本存在：

```text
metric_calculation/result_test.py
```

但脚本内部的 `folder_path` 当前硬编码为：

```python
folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_taxi_final'
```

评估完成后需要把它改成实际输出目录：

```python
folder_path = 'result_test/cross-region/NYC_taxi'
```

## 2. 云服务器环境要求

推荐配置：

```text
OS: Ubuntu 20.04 或 Ubuntu 22.04
Python: 3.9.13
CUDA: 11.7
GPU: 推荐 8 x A100；只评估时可按显存情况减少 GPU 数
```

如果云服务器 GPU 数不是 8，需要修改：

```bash
num_gpus=8
```

改成实际 GPU 数，例如：

```bash
num_gpus=4
```

## 3. 上传项目到云服务器

建议把当前整个项目目录上传到云服务器，例如：

```bash
scp -r UrbanGPT user@server:/data/
```

云端进入项目目录：

```bash
cd /data/UrbanGPT
```

确认关键目录：

```bash
ls checkpoints
ls ST_data_urbangpt/NYC_taxi_cross-region
ls urbangpt/eval
```

## 4. 创建 Conda 环境

```bash
conda create -n urbangpt python=3.9.13 -y
conda activate urbangpt
```

安装 PyTorch CUDA 11.7 版本：

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

安装基础依赖：

```bash
pip install "fschat[model_worker,webui]==0.2.34"
pip install transformers==4.34.0
pip install deepspeed==0.12.6
pip install ray==2.9.0
pip install einops==0.7.0
pip install wandb==0.15.11
pip install peft==0.7.1
pip install -r requirements.txt
```

安装 PyG 相关依赖：

```bash
pip install torch_geometric==2.4.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

安装 flash-attn：

```bash
pip install flash-attn==2.3.5
```

如果 `flash-attn` 编译失败，建议下载与 CUDA 11.7、Torch 2.0.1、Python 3.9 匹配的预编译 wheel 后安装。

## 5. 下载官方 UrbanGPT checkpoint

官方模型地址：

```text
https://huggingface.co/bjdwh/UrbanGPT
```

下载到本项目推荐目录：

```bash
huggingface-cli download bjdwh/UrbanGPT --local-dir ./checkpoints/UrbanGPT_pretrained
```

如果访问 HuggingFace 较慢，可先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bjdwh/UrbanGPT --local-dir ./checkpoints/UrbanGPT_pretrained
```

下载后检查：

```bash
ls -lh checkpoints/UrbanGPT_pretrained
```

该目录不能是空目录，应包含模型配置、权重、tokenizer 或 adapter 相关文件。

## 6. 检查评估脚本

当前 `urbangpt_eval.sh` 应保持如下配置：

```bash
output_model=./checkpoints/UrbanGPT_pretrained
datapath=./ST_data_urbangpt/NYC_taxi_cross-region/NYCtaxi_zeroshot.json
st_data_path=./ST_data_urbangpt/NYC_taxi_cross-region/NYCtaxi_zeroshot_pkl.pkl
res_path=./result_test/cross-region/NYC_taxi
start_id=0
end_id=51920
num_gpus=8
```

如果 GPU 数不是 8，修改：

```bash
num_gpus=实际GPU数量
```

## 7. 运行环境检查

在云服务器上执行：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import transformers; print(transformers.__version__)"
python -c "from urbangpt.model.ST_Llama import STLlamaForCausalLM; print('UrbanGPT import OK')"
```

预期：

```text
torch.cuda.is_available() 为 True
transformers 版本为 4.34.0
UrbanGPT import OK
```

## 8. 运行官方模型评估

```bash
conda activate urbangpt
cd /data/UrbanGPT
bash urbangpt_eval.sh
```

评估结果输出目录：

```text
result_test/cross-region/NYC_taxi
```

输出文件形如：

```text
arxiv_test_res_0_6490.json
arxiv_test_res_6490_12980.json
...
```

具体分片数量取决于 `num_gpus`。

## 9. 计算评估指标

评估完成后，修改：

```text
metric_calculation/result_test.py
```

将：

```python
folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_taxi_final'
```

改成：

```python
folder_path = 'result_test/cross-region/NYC_taxi'
```

保持：

```python
mode = 'regression'
```

然后运行：

```bash
cd /data/UrbanGPT/metric_calculation
python result_test.py
```

注意：如果从 `metric_calculation` 目录运行，`folder_path` 应写成相对该目录的路径：

```python
folder_path = '../result_test/cross-region/NYC_taxi'
```

也可以从项目根目录运行：

```bash
cd /data/UrbanGPT
python metric_calculation/result_test.py
```

此时 `folder_path` 使用：

```python
folder_path = 'result_test/cross-region/NYC_taxi'
```

## 10. 加入可解释性机制的推荐路线

当前不建议重新训练 UrbanGPT。推荐采用 post-hoc explanation：

1. 保留 UrbanGPT 原始预测流程。
2. 使用 `st_pre_infolow` 和 `st_pre_outfolow` 作为最终预测值。
3. 从结果中读取：
   - `x_in`
   - `x_out`
   - `st_pre_infolow`
   - `st_pre_outfolow`
   - 原始 prompt 中的时间、区域、POI 信息
4. 构造解释提示词，让 LLM 解释预测趋势。
5. 将解释写回结果 JSON。

推荐不要直接改原始 `run_urbangpt.py`，而是复制一个新脚本：

```bash
cp urbangpt/eval/run_urbangpt.py urbangpt/eval/run_urbangpt_explain.py
```

解释插入点位于预测结果生成之后：

```python
st_pre_infolow = st_pre_final[:, :, :, 0].squeeze().detach().cpu().tolist()
st_pre_outfolow = st_pre_final[:, :, :, 1].squeeze().detach().cpu().tolist()
```

在写入 `res_data.append(...)` 时新增字段：

```python
"explanation": explanation,
"explanation_type": "post_hoc_cot"
```

解释 prompt 可采用：

```text
Given the historical taxi inflow and outflow, the regional POI information, the time period, and the model prediction,
explain why the predicted inflow and outflow may increase, decrease, or remain stable.

Please analyze:
1. Recent temporal trend
2. Inflow-outflow imbalance
3. Time-of-day and weekday effects
4. Regional POI influence
5. A concise explanation for the predicted result

Do not change the predicted values.
```

## 11. 推荐执行顺序

第一阶段：官方模型评估

```text
1. 云服务器创建 conda 环境
2. 安装 CUDA 版本 PyTorch 和依赖
3. 下载 bjdwh/UrbanGPT 到 checkpoints/UrbanGPT_pretrained
4. 检查 urbangpt_eval.sh
5. 运行 bash urbangpt_eval.sh
6. 修改并运行 metric_calculation/result_test.py
7. 保存官方模型指标作为 baseline
```

第二阶段：加入解释机制

```text
1. 复制 run_urbangpt.py 为 run_urbangpt_explain.py
2. 在预测值生成后构造 explanation prompt
3. 调用 LLM 生成解释
4. 输出到 result_test/cross-region/NYC_taxi_explain
5. 保持预测值不变，仅新增 explanation 字段
6. 对比原始结果与解释增强结果
```

## 12. 当前最重要的下一步

当前本地目录唯一明显缺口是：

```text
checkpoints/UrbanGPT_pretrained/ 为空
```

下一步请先在云服务器或本地执行：

```bash
huggingface-cli download bjdwh/UrbanGPT --local-dir ./checkpoints/UrbanGPT_pretrained
```

然后再运行：

```bash
bash urbangpt_eval.sh
```
