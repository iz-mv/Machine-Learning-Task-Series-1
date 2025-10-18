from pathlib import Path
from datetime import datetime
from llama_cpp import Llama
import sys
import io

# фиксим stdin/stdout под macOS + PyCharm
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "data" / "model.gguf"
    if not model_path.exists():
        print(f"Нет модели: {model_path}\nПоложи сюда GGUF (Llama-3.2-3B-Instruct Q4_K_M) и назови model.gguf")
        return

    #настройки под мой ноут air m2
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048, #контекст (если будет тесно по памяти - 1536)
        n_gpu_layers=-1, #полное оффлоад-ускорение через Metal, если не m mac, то можете 0 поставить (вроде?)
        n_batch=192, #размер батча при генерации (128–256)
        seed=42
    )

    system_prompt = (
        "You are a helpful, concise assistant. "
        "Answer briefly unless the user asks for details. "
        "Use bullet points when listing."
    )

    history = [{"role": "system", "content": system_prompt}]
    print("Локальный чат с Llama-3.2-3B-Instruct. Напишите 'exit' чтобы выйти")

    #отчет сессии будет в outputs
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"llm_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("=== LLM local chat session ===\n")

        while True:
            user = input("\nТы: ").strip()
            if user.lower() in {"exit", "quit"}:
                print("Пока!")
                break

            history.append({"role": "user", "content": user})
            log.write(f"\n[USER]\n{user}\n")

            #cтриминговый вывод ответа(чтобы видеть текст по мере генерации)
            print("LLM: ", end="", flush=True)
            chunks = llm.create_chat_completion(
                messages=history,
                temperature=0.3, #креативность (можно подредактировать)
                max_tokens=512, #длина ответа
                stream=True
            )

            reply_parts = []
            for ch in chunks:
                delta = ch["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    reply_parts.append(delta)
            print()  #перенос строки после ответа

            reply = "".join(reply_parts).strip()
            history.append({"role": "assistant", "content": reply})
            log.write(f"\n[ASSISTANT]\n{reply}\n")

if __name__ == "__main__":
    main()
