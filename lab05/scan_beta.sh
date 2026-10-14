set -e
cd "$(dirname "$0")"  # 可移植：脚本所在目录
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}   # 可移植：可用环境变量覆盖（如 PY=/d/.../venv/Scripts/python.exe）
for b in 0.05 0.5; do
  echo "########## beta=$b ##########"
  rm -rf dpo_adapter
  $PY hw5_dpo.py --mode dpo --beta $b > logs_dpo_b$b.txt 2>&1
  grep -E "\[dpo\] ep" logs_dpo_b$b.txt | sed "s/^/  [b=$b] /"
  cp dpo_log.csv dpo_log_b$b.csv
  cp -r dpo_adapter dpo_adapter_b$b
  $PY - <<PYEOF > logs_eval_b$b.txt 2>&1
import importlib.util, torch
spec = importlib.util.spec_from_file_location('m','hw5_dpo.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
proc = AutoProcessor.from_pretrained(m.resolve_model_path())
base = AutoModelForImageTextToText.from_pretrained(m.resolve_model_path(), dtype=torch.float32).to("cuda")
mdl = PeftModel.from_pretrained(base, "dpo_adapter_b$b").to("cuda")
tot = 0
for tp in ("count","attr","exist"):
    qs=[q for q in m.QUESTIONS if q["type"]==tp]
    ok=sum(m.judge(q, m.answer(proc, mdl, m.load_image(q["img"]), q["q"])) for q in qs)
    tot += ok
    print(f"RESULT b=$b {tp} {ok}/{len(qs)}")
print(f"RESULT b=$b TOTAL {tot}/15")
PYEOF
  grep "^RESULT" logs_eval_b$b.txt
done
rm -rf dpo_adapter && cp -r dpo_adapter_r8 dpo_adapter && cp dpo_log.csv dpo_log_b0.1.csv 2>/dev/null || true
echo "########## 完成，已恢复 beta=0.1 ##########"
