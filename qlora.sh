#python qlora.py --model_name_or_path /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b --batch_size=8 --do_eval=false \
# --do_train=true --omni_eval=true --dataset wikitext --awq=false --bits=4 --max_steps 2500 --bnb=true --learning_rate=0.00005

d=/data/shared/CompressaAI/LLaMA/llama-1_2-7_13b

for MODEL in llama-7b llama-13b Llama-2-7b-hf Llama-2-13b-hf
do
echo $MODEL
python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend real \
 --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=4 --max_steps 2500 --bnb=false --learning_rate=0.00005 --load_quant=$d/quant_cache/$MODEL-w4-g128-awq.pt --q_group_size 128
done

# --load_quant=/home/prutko_a/work/AWQ/quant_cache/llama-7b-w4-g128-awq.pt --dataset alpaca --max_steps 100 --bits=4 --q_backend real