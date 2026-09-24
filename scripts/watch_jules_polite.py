#!/usr/bin/env python3
"""
控えめポーリング版 Jules 監視スクリプト (watch_jules_polite.py)

頻繁な API チェックによるレート制限や過負荷を防止するため、
デフォルト 120 秒（2分）間隔でチェックし、指数バックオフや進行状況のログ出力を行います。
"""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

def run_jules_list() -> str:
    env = os.environ.copy()
    env["COLUMNS"] = "500"
    try:
        res = subprocess.run(
            ["jules", "remote", "list", "--session"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return res.stdout
    except Exception as e:
        raise RuntimeError(f"jules remote list 実行失敗: {e}")

def parse_sessions(output: str) -> List[Dict[str, str]]:
    sessions = []
    lines = output.strip().split("\n")
    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("ID"):
            continue
        match = re.match(r"^\s*(\d+)\s+(.+)$", line)
        if not match:
            continue
        session_id = match.group(1)
        rest = match.group(2)
        
        status = "In Progress"
        if "Completed" in rest or "Comp" in rest:
            status = "Completed"
        elif "Awaiting" in rest or "Awai" in rest:
            status = "Awaiting User Feedback"
        elif "Failed" in rest:
            status = "Failed"

        repo_match = re.search(r"([\w\-]+/[\w\-\.]+)", rest)
        repo = repo_match.group(1) if repo_match else "unknown"

        sessions.append({
            "id": session_id,
            "status": status,
            "repo": repo,
            "raw": line_clean,
        })
    return sessions

def main():
    parser = argparse.ArgumentParser(description="Jules 控えめ監視スクリプト")
    parser.add_argument("--session", help="監視するセッションID")
    parser.add_argument("--repo", default="chottokun/logit-rerank-rag", help="監視するリポジトリ")
    parser.add_argument("--interval", type=int, default=120, help="チェック間隔（秒、デフォルト: 120秒 = 2分）")
    parser.add_argument("--auto-pull", action="store_true", help="完了時に自動 pull")
    args = parser.parse_args()

    print("==================================================")
    print(" Jules 監視スクリプトを開始します")
    print(f" - 対象セッション: {args.session or '最新セッションを自動検出'}")
    print(f" - 対象リポジトリ: {args.repo}")
    print(f" - チェック間隔: {args.interval} 秒（頻繁なアクセスを回避）")
    print("==================================================")

    last_status = None
    while True:
        try:
            raw_out = run_jules_list()
            sessions = parse_sessions(raw_out)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] リスト取得待機中: {e}")
            time.sleep(args.interval)
            continue

        target = None
        if args.session:
            target = next((s for s in sessions if s["id"] == args.session), None)
        else:
            # 指定リポジトリの最新セッション
            repo_sessions = [s for s in sessions if args.repo in s["repo"] or "logit-rerank" in s["raw"]]
            if repo_sessions:
                target = repo_sessions[0]

        if not target:
            print(f"[{time.strftime('%H:%M:%S')}] 対象セッション待機中...")
        else:
            sid = target["id"]
            status = target["status"]
            if status != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] [状態変化] セッション {sid}: 【{status}】")
                last_status = status
            else:
                print(f"[{time.strftime('%H:%M:%S')}] セッション {sid} は進行中... 次のチェックまで {args.interval} 秒待機")

            if status == "Completed":
                print(f"\n🎉 セッション {sid} が完了 (Completed) しました！")
                if args.auto_pull:
                    print("手元への pull を実行します...")
                    subprocess.run(["jules", "remote", "pull", "--session", sid])
                sys.exit(0)
            elif status == "Awaiting User Feedback":
                print(f"\n🔔 セッション {sid} がユーザーの入力待ち (Awaiting User Feedback) になりました。")
                print(f"   `jules teleport {sid}` で確認してください。")
                sys.exit(10)
            elif status == "Failed":
                print(f"\n❌ セッション {sid} が失敗 (Failed) しました。")
                sys.exit(1)

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
