cd "$(dirname "$0")"  # 可移植：脚本所在目录
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}   # 可移植：可用环境变量覆盖（如 PY=/d/.../venv/Scripts/python.exe）
echo "###TIMING_START###"
for step in "infer:--mode infer" "sft:--mode sft" "dpo:--mode dpo"; do
  name="${step%%:*}"; arg="${step##*:}"
  s=$(date +%s)
  $PY hw5_dpo.py $arg > /tmp/t_$name.log 2>&1
  e=$(date +%s)
  echo "TIMING $name $((e-s)) 秒"
done
s=$(date +%s)
$PY hw5_dpo.py --mode eval > /tmp/t_eval.log 2>&1
e=$(date +%s)
echo "TIMING eval $((e-s)) 秒"
echo "###TIMING_END###"
