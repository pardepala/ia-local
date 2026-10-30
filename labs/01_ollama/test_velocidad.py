import urllib.request
import json
import time

prompt = "Dame un consejo de 1 sola frase inspiradora para un Ingeniero en Informática."
payload = json.dumps({"model": "qwen2.5:7b", "prompt": prompt, "stream": False}).encode("utf-8")

req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))

eval_count = res.get("eval_count", 0)
eval_duration_sec = res.get("eval_duration", 1) / 1e9
tokens_per_sec = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0

print("--- RESPUESTA DE QWEN 2.5 (7B) EN LOCAL ---")
print(res.get("response"))
print("-------------------------------------------")
print(f"Tokens generados: {eval_count}")
print(f"Velocidad real de tu RTX 4060: {tokens_per_sec:.1f} tokens/segundo")
