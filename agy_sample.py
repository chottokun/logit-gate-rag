import asyncio
import os
import glob
from google import genai
from google.antigravity import Agent, LocalOpenAIAgentConfig
from google.antigravity.hooks import policy

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PLAN_DIR = os.path.join(PROJECT_ROOT, "plan")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
os.makedirs(SRC_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Cloud Architect: 計画書を解析してタスクリストを生成
# --------------------------------------------------
def analyze_plans_with_cloud() -> list[str]:
    print("Cloud Architect (Gemini) が計画書を分析中...")
    
    # plan 配下のすべての Markdown を収集
    plan_texts = []
    for filepath in sorted(glob.glob(os.path.join(PLAN_DIR, "*.md"))):
        with open(filepath, "r", encoding="utf-8") as f:
            plan_texts.append(f"--- File: {os.path.basename(filepath)} ---\n{f.read()}")
    
    full_context = "\n\n".join(plan_texts)

    client = genai.Client()
    prompt = f"""
あなたはソフトウェアアーキテクトです。
以下の仕様書・計画書を読み込み、ローカルワーカー（Gemma 4）に順次実行させる「具体的で独立した実装タスクの指示文リスト」を作成してください。

【制約】
- 出力はPythonが eval() または行分割でそのままパースできるよう、1行につき1つの具体的なタスク指示を出力してください（プレフィックスとして 'TASK: ' を付ける）。
- タスクごとに「作成・修正する絶対パス」「実装すべき関数やクラスの要件」を明確に含めてください。
- 作業ディレクトリのルート: {PROJECT_ROOT}
- ソースコード出力先: {SRC_DIR}

【計画書・リファレンス】
{full_context}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash", # または利用可能な最新 Flash モデル
        contents=prompt,
    )
    
    tasks = [
        line.replace("TASK:", "").strip() 
        for line in response.text.split("\n") 
        if line.strip().startswith("TASK:")
    ]
    return tasks

# --------------------------------------------------
# 2. On-Device Builder: 各タスクをローカルGPUで実装・テスト
# --------------------------------------------------
local_config = LocalOpenAIAgentConfig(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="gemma4:latest",
    workspaces=[PROJECT_ROOT],
    policies=[policy.allow_all()]
)

async def execute_task_locally(task_instruction: str, step_num: int, total_steps: int):
    print(f"\n==========================================")
    print(f"[{step_num}/{total_steps}] Local Gemma 4 実行開始")
    print(f"指示: {task_instruction}")
    print(f"==========================================\n")

    prompt = f"""
あなたはローカル実装ワーカーです。以下のタスクを忠実に実行してください。

【厳守ルール】
- ツール呼び出しで指定するファイルパスは、例外なく完全な絶対パス（{PROJECT_ROOT}/...）を使用してください。相対パスは禁止です。

【タスク】
{task_instruction}

実装完了後、ファイルが存在することを確認し、可能であれば簡易テストや syntax check を実行してください。
"""
    async with Agent(local_config) as agent:
        response = await agent.chat(prompt)
        async for token in response:
            print(token, end="", flush=True)

# --------------------------------------------------
# 3. メインオーケストレーション
# --------------------------------------------------
async def main():
    # クラウドで計画分解
    tasks = analyze_plans_with_cloud()
    print(f"\n分解されたタスク数: {len(tasks)}")
    for i, t in enumerate(tasks, 1):
        print(f"  {i}. {t}")

    # ローカルで1タスクずつ順次実装
    for i, task in enumerate(tasks, 1):
        await execute_task_locally(task, i, len(tasks))

if __name__ == "__main__":
    asyncio.run(main())