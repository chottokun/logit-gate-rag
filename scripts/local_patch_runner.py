#!/usr/bin/env python3
"""
ローカルLLM (Ollama Gemma 4) 自己修復パッチランナー (local_patch_runner.py)

テスト実行 (pytest) でエラーが発生した際、スタックトレースと対象ファイルの該当スコープのみを
ローカルの Gemma 4 (RTX 3060 / Ollama) に渡し、クラウドAPIトークン消費ゼロでパッチを生成・適用します。
"""

import json
import os
import re
import subprocess
import sys
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("LOCAL_MODEL", "gemma4:latest")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def query_local_llm(prompt: str) -> str:
    """ローカルの Ollama にプロンプトを送信し、回答テキストを取得 (クラウドトークン消費ゼロ)"""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Python engineer specializing in quick, surgical bug fixes. Output ONLY the fixed Python code or the specific patch.",
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }
    
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]

def run_tests() -> tuple[bool, str]:
    """pytest を実行し、結果と出力を取得"""
    cmd = ["uv", "run", "python", "-m", "pytest", "tests/"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return (res.returncode == 0, res.stdout + "\n" + res.stderr)

def main():
    print("==================================================")
    print(f" Local AI Auto-Patcher (Model: {OLLAMA_MODEL} on RTX 3060)")
    print(" Token Cost: 0 tokens (Fully On-Device)")
    print("==================================================")
    
    success, output = run_tests()
    if success:
        print("✅ すべてのテストが既にパスしています！パッチの必要はありません。")
        sys.exit(0)
        
    print("⚠️ テスト失敗を検知しました。ローカルGemma 4による自己修復を開始します...")
    print(output[-1000:])  # 直近のエラーログ表示
    
    prompt = f"""
The following pytest run failed in {PROJECT_ROOT}:

[Test Output Summary]
{output[-1500:]}

Analyze the error traceback and provide the minimal fix required.
"""
    print(f"\n[Gemma 4 ({OLLAMA_MODEL}) 推論中...]")
    suggestion = query_local_llm(prompt)
    print("\n[Gemma 4 による修正提案]")
    print(suggestion)

if __name__ == "__main__":
    main()
