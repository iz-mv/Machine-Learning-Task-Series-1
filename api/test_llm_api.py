import requests

payload = {
  "messages": [
    {"role":"system","content":"You are a helpful assistant."},
    {"role":"user","content":"Give me 3 ideas for a weekend project with Python."}
  ],
  "temperature": 0.3,
  "max_tokens": 200
}
r = requests.post("http://127.0.0.1:8000/chat", json=payload, timeout=180)
print(r.json())
