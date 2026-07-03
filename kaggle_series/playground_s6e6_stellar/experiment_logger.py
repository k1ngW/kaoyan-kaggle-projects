"""
简易实验日志：记录每次运行的 OOF、LB、参数
"""
import json, os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "experiment_logs"
LOG_DIR.mkdir(exist_ok=True)

def log_experiment(version: str, oof_scores: dict, lb_score: float = None,
                   params: dict = None, notes: str = ""):
    """记录一次实验"""
    entry = {
        "version": version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "oof_scores": oof_scores,
        "lb_score": lb_score,
        "params": params or {},
        "notes": notes,
    }
    # 追加到总日志
    log_file = LOG_DIR / "all_experiments.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # 单独保存一份
    single_file = LOG_DIR / f"{version}_{datetime.now().strftime('%m%d_%H%M')}.json"
    with open(single_file, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    print(f"[LOG] Experiment saved: {single_file}")

def show_all_experiments():
    """展示所有历史实验"""
    log_file = LOG_DIR / "all_experiments.jsonl"
    if not log_file.exists():
        print("暂无实验记录")
        return
    print(f"{'Ver':<6} {'Time':<20} {'OOF':<10} {'LB':<10} {'Notes'}")
    print("-" * 70)
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            oof_str = str(e.get("oof_scores", {}))
            lb_str = f"{e['lb_score']:.5f}" if e.get("lb_score") else "N/A"
            print(f"{e['version']:<6} {e['timestamp']:<20} {oof_str:<10} {lb_str:<10} {e.get('notes','')[:30]}")

if __name__ == "__main__":
    show_all_experiments()
