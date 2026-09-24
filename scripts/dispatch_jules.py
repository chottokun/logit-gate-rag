#!/usr/bin/env python3
"""
Jules タスク投入スクリプト (dispatch_jules.py)

./plan/search_logit_sem.md の仕様に基づき、Jules にリポジトリ全容のインデックスと
コンポーネント実装（PR生成）を指示するセッションを開始します。
"""

import subprocess
import sys
import re

PROMPT = """You are a senior software architect and machine learning engineer.
Please index the repository and implement the core components specified in `plan/search_logit_sem.md` and `plan/ref.md`.

### Key Tasks:
1. Implement the lightweight semantic search module (`src/search/`):
   - Ruri-v3 embedding provider with pure SentencePiece (unigram) support.
   - Strictly follow asymmetric prefixes: "検索クエリ: " for queries and "検索文書: " for passages.
2. Implement the direct pointwise logit router (`src/router/`):
   - Qwen2.5 pointwise logit extraction at LM-Head (without autoregressive token generation).
   - Evaluate "Yes" (token ID: 9693) and "No" (token ID: 2154) following `<|im_start|>assistant\\n`.
   - Calculate sigmoid score margin with temperature parameter.
3. Integrate the pipeline (`src/pipeline/`):
   - Dense retrieval candidate generation -> Logit threshold filtering.
4. Add comprehensive unit tests in `tests/` covering:
   - Prefix correctness.
   - Logit extraction shape and score calculations.
   - End-to-end pipeline filtering.

Please create a clean branch and open a Pull Request when completed.
"""

def main():
    print("==================================================")
    print(" Dispatching implementation task to Google Jules...")
    print(" Target repo: chottokun/logit-gate-rag")
    print("==================================================")

    cmd = ["jules", "remote", "new", "--session", PROMPT]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
        
        # セッションIDを抽出 (例: Session created: 1234567890 または出力中の数字列)
        m = re.search(r"(\d{15,})", res.stdout)
        session_id = m.group(1) if m else None
        if session_id:
            print(f"\n[SUCCESS] Jules Session ID: {session_id}")
            print(f"To monitor progress:")
            print(f"python3 scripts/watch_jules_polite.py --session {session_id}")
            return session_id
        else:
            print("[INFO] Session started. Check status with `jules remote list --session`.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to start Jules session: {e.stderr}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
