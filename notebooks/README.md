# Обучение адаптеров для квантованных моделей
## 1. Установка зависимостей

Для выполнения обучения адаптеров LoRA по примеру из ноутбуков нужно установить необходимые зависимости

```bash
conda create --name llm python=3.10 -y
conda activate llm
conda install git git-lfs -y
git lfs install

# cudatoolkit-dev
conda install -c conda-forge cudatoolkit-dev cudnn -y
export CUDA_HOME=$CONDA_PREFIX

# torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers accelerate bitsandbytes scipy sentencepiece timm==0.5.4 einops evaluate wandb

# llama.cpp
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
sed -i 's/;70/;70;80/' vendor/llama.cpp/CMakeLists.txt
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -e . --force-reinstall --upgrade --no-cache-dir
cd ..

# llm-awq
git clone --single-branch --branch dev git@github.com:compressa-ai/llm-awq.git
cd llm-awq
pip install -e .
cd awq/kernels
python setup.py install
cd ../../..

# smoothquant
git clone --single-branch --branch dev git@github.com:compressa-ai/smoothquant.git
cd smoothquant
python setup.py install
cd ..

# peft
git clone --single-branch --branch dev git@github.com:compressa-ai/peft.git
cd peft
pip install -e .
cd ..

# qlora
git clone --single-branch --branch dev git@github.com:compressa-ai/qlora.git

# OmniQuant
git clone git@github.com:compressa-ai/OmniQuant.git

# vllm
git clone --single-branch --branch dev git@github.com:compressa-ai/vllm.git
pip install vllm

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib
```

## 2. Квантизация моделей

На текущий момент поддерживается 3 типа квантизации

- ### [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)  

bitsandbytes работает из коробки, при запуске обучения адаптеров нужно просто указать в какой битности загружать модель

- ### [AWQ](https://github.com/compressa-ai/llm-awq/tree/dev)  

Пример квантизации модели

```bash
cd llm-awq
MODEL=llama-7b

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path /dataset/llama-hf/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path /dataset/llama-hf/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake

# generate real quantized weights (w4)
python -m awq.entry --model_path /dataset/llama-hf/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w4-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
python -m awq.entry --model_path /dataset/llama-hf/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant quant_cache/$MODEL-w4-g128-awq.pt
```
Итоговый чекпоинт, с которым нужно будет обучать адаптеры - `quant_cache/$MODEL-w4-g128-awq.pt`

- ### [OmniQuant](https://github.com/compressa-ai/OmniQuant/tree/main)  

Пример квантизации модели

```bash
cd OmniQuant
MODEL=llama-7b

# Training the quantization parameters.
python main.py \
--model /dataset/llama-hf/$MODEL \
--epochs 40 --output_dir ./log/$MODEL-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss \
--nsamples 32

# Saving the really quantized model.
python main.py \
--model /dataset/llama-hf/$MODEL \
--epochs 0  --output_dir ./log/temp \
--wbits 3 --abits 16 --group_size 128 --nsamples 32 \
--resume ./log/$MODEL-w3a16g128/omni_parameters.pth \
--real_quant --save_dir /dataset/llama-hf/$MODEL-oq
```

**! Сейчас пока поддерживается обучение адаптеров LoRA только при загрузке fake quantized  модели!**

## 3. Обучение адаптеров LoRA

- ### bitsandbytes
```bash
cd qlora
MODEL=llama-7b

python qlora.py --model_name_or_path /dataset/llama-hf/$MODEL --batch_size=8 \
  --do_eval=false --do_train=true --omni_eval=true --dataset wikitext \
  --bnb=true --bits=4 --max_steps 2500 --learning_rate=0.00001 \
  --lora_r=4 --qlora=true
```

- ### AWQ

```bash
cd qlora
MODEL=llama-7b

python qlora.py --model_name_or_path /dataset/llama-hf/$MODEL --batch_size=8 \
  --do_eval=false --q_backend real --do_train=true --omni_eval=true \
  --dataset wikitext --awq=true --bits=4 --max_steps 2500 --learning_rate=0.00001 \
  --load_quant=quant_cache/$MODEL-w4-g128-awq.pt --q_group_size 128 \
  --lora_r=4 --qlora=true
```

- ### OmniQuant

```bash
cd qlora
MODEL=llama-7b

python qlora.py --model_name_or_path /dataset/llama-hf/$MODEL --batch_size=8 \
  --omniquant omniquant_checkpoint --do_eval=false --do_train=true --omni_eval=true \
  --dataset wikitext --bits=4 --max_steps 2500 --learning_rate=0.00001 \
  --lora_r=4 --qlora=true
```