#python qlora.py --model_name_or_path /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b --batch_size=8 --do_eval=false \
# --do_train=true --omni_eval=true --dataset wikitext --awq=false --bits=4 --max_steps 2500 --bnb=true --learning_rate=0.00005

d=/data/shared/CompressaAI/LLaMA/llama-1_2-7_13b
MODEL=llama-7b
pc=/data/shared/CompressaAI/experiments/clean
g=64
lr=0.000002

for MODEL in llama-7b # llama-7b llama-13b Llama-2-7b-hf Llama-2-13b-hf
# for oq in `ls /data/shared/CompressaAI/experiments/clean`
do
echo $MODEL
# echo $oq

# # fp ln
# python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend real \
#  --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=0 --max_steps 2500 \
#  --bnb=false --learning_rate=$lr --load_quant=$d/quant_cache/$MODEL-w4-g$g-awq.pt --q_group_size $g \
#  --lora_r=4 --qlora=false --ln=true --bias=false

#  # fp bias
# python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend real \
#  --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=0 --max_steps 2500 \
#  --bnb=false --learning_rate=$lr --load_quant=$d/quant_cache/$MODEL-w4-g$g-awq.pt --q_group_size $g \
#  --lora_r=4 --qlora=false --ln=false --bias=true

# awq qlora ln bias
python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend real \
 --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=4 --max_steps 2500 \
 --bnb=false --learning_rate=$lr --load_quant=$d/quant_cache/$MODEL-w4-g$g-awq.pt --q_group_size $g \
 --lora_r=4 --qlora=true --ln=true --biases=true --output_dir=out/all

# # awq qlora
# python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend real \
#  --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=4 --max_steps 2500 \
#  --bnb=false --learning_rate=$lr --load_quant=$d/quant_cache/$MODEL-w4-g$g-awq.pt --q_group_size $g \
#  --lora_r=4 --qlora=true --ln=false --bias=false # --load_ln=out/ln.pt --output_dir=out/awq_4b_g64_ln_8e-5_qlora

# # awq ln
# python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend real \
#  --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=4 --max_steps 2500 \
#  --bnb=false --learning_rate=$lr --load_quant=$d/quant_cache/$MODEL-w4-g$g-awq.pt --q_group_size $g \
#  --lora_r=4 --qlora=false --ln=true --bias=false --output_dir=out/awq_4b_g64_ln_8e-5 

# # awq bias
#  python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend real \
#  --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=4 --max_steps 2500 \
#  --bnb=false --learning_rate=$lr --load_quant=$d/quant_cache/$MODEL-w4-g$g-awq.pt --q_group_size $g \
#  --lora_r=4 --qlora=false --ln=false --biases=true --output_dir=out/bias

# # awq ln bias
#   python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend real \
#  --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=4 --max_steps 2500 \
#  --bnb=false --learning_rate=$lr --load_quant=$d/quant_cache/$MODEL-w4-g$g-awq.pt --q_group_size $g \
#  --lora_r=4 --qlora=false --ln=true --bias=true

# python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --q_backend fake \
#  --do_train=true --omni_eval=true --dataset wikitext --awq=true --bits=3 --max_steps 2500 --bnb=false \
#   --learning_rate=$lr --load_awq=$d/awq_cache/$MODEL-w3-g$g.pt --q_group_size $g --lora_r=4

# # bnb qlora
# python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --do_train=true \
#   --omni_eval=true --dataset wikitext --bnb=true --awq=false --bits=4 --max_steps 2500 --learning_rate=$lr \
#   --lora_r=4 --qlora=true --ln=false --bias=false

# # bnb ln
# python qlora.py --model_name_or_path $d/$MODEL --batch_size=8 --do_eval=false --do_train=true \
#   --omni_eval=true --dataset wikitext --bnb=true --awq=false --bits=4 --max_steps 2500 --learning_rate=$lr \
#   --lora_r=4 --qlora=false --ln=true --bias=false

# # oq ln
# python qlora.py --model_name_or_path $d/$MODEL --omniquant $pc/$oq/ckpt \
#   --batch_size=8 --do_eval=false --do_train=true --qlora=false --ln=false --bias=false \
#   --omni_eval=true --dataset wikitext --bnb=false --awq=false --bits=4 --max_steps 2500 --learning_rate=$lr --lora_r=4

done

# --load_quant=/home/prutko_a/work/AWQ/quant_cache/llama-7b-w4-g128-awq.pt --dataset alpaca --max_steps 100 --bits=4 --q_backend real