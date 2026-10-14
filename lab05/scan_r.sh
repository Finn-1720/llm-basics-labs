set -e
cd "$(dirname "$0")"  # 可移植：脚本所在目录
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}   # 可移植：可用环境变量覆盖（如 PY=/d/.../venv/Scripts/python.exe）
for r in 4 16; do
  echo "########## r=$r ##########"
  rm -rf sft_adapter
  $PY hw5_dpo.py --mode sft --r $r > logs_sft_r$r.txt 2>&1
  grep -E "trainable params" logs_sft_r$r.txt | sed "s/^/  [r=$r] /"
  grep -E "\[sft\] ep" logs_sft_r$r.txt | sed "s/^/  [r=$r] /"
  cp -r sft_adapter sft_adapter_r$r
  # 评该 r 下的 SFT 正确率
  $PY - <<PYEOF > logs_eval_r$r.txt 2>&1
import importlib.util, torch
spec = importlib.util.spec_from_file_location('m','hw5_dpo.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
proc = AutoProcessor.from_pretrained(m.resolve_model_path())
base = AutoModelForImageTextToText.from_pretrained(m.resolve_model_path(), dtype=torch.float32).to("cuda")
mdl = PeftModel.from_pretrained(base, "sft_adapter_r$r").to("cuda")
for tp in ("count","attr","exist"):
    qs=[q for q in m.QUESTIONS if q["type"]==tp]
    ok=sum(m.judge(q, m.answer(proc, mdl, m.load_image(q["img"]), q["q"])) for q in qs)
    print(f"RESULT r=$r {tp} {ok}/{len(qs)}")
PYEOF
  grep "^RESULT" logs_eval_r$r.txt
done
rm -rf sft_adapter && cp -r sft_adapter_r8 sft_adapter
echo "########## 完成，已恢复 r=8 ##########"
