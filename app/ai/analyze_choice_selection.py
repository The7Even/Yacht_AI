"""FastEV first-turn Choice selection analysis."""
from __future__ import annotations
import argparse, csv, json, os, random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.scoring import ScoreCalculator
from .action_generator import ActionType
from .expected_value_strategy import ExpectedValueStrategy

FIELDS=("sample","first_dice","choice_score","selected_category","selected_score","best_category","best_category_score","score_gap_choice_vs_best","choice_is_best","choice_selected","choice_selected_when_not_best")
_WORKER_STRATEGY=None

def _worker_init():
    global _WORKER_STRATEGY
    _WORKER_STRATEGY=ExpectedValueStrategy()

def _observe(payload):
    sample,seed=payload; strategy=_WORKER_STRATEGY or ExpectedValueStrategy()
    engine=GameEngine(dice_roller=DiceRoller(random.Random(seed))); engine.start_game(); engine.roll_dice()
    while True:
        action=strategy.decide(engine.state).action
        if action.type is ActionType.SCORE:
            selected=action.selected_category; assert selected is not None
            dice=tuple(engine.state.current_dice or ()); break
        desired=frozenset(action.held_indices)
        for i in engine.state.held_indices-desired: engine.unhold_dice(i)
        for i in desired-engine.state.held_indices: engine.hold_dice(i)
        engine.roll_dice()
    scores={c:ScoreCalculator.calculate(c,dice) for c in ALL_CATEGORIES}
    best=max(scores,key=scores.get); choice=scores[Category.CHOICE]
    return {"sample":sample,"first_dice":json.dumps(list(dice)),"choice_score":choice,"selected_category":selected.name,"selected_score":scores[selected],"best_category":best.name,"best_category_score":scores[best],"score_gap_choice_vs_best":choice-scores[best],"choice_is_best":int(selected is best),"choice_selected":int(selected is Category.CHOICE),"choice_selected_when_not_best":int(selected is Category.CHOICE and selected is not best)}

def _progress(done,total,workers):
    pct=done/total*100 if total else 100; width=40; filled=int(width*pct/100)
    print(f"\rChoice Selection [{'#'*filled+'.'*(width-filled)}] {pct:6.2f}% | {done}/{total} first turns | {workers} workers",end="",flush=True)

def _summary(path,rows):
    total=len(rows); selected=Counter(r["selected_category"] for r in rows); choice=sum(r["choice_selected"] for r in rows); nonbest=sum(r["choice_selected_when_not_best"] for r in rows); best=sum(r["choice_is_best"] for r in rows); gaps=[r["score_gap_choice_vs_best"] for r in rows]
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(["metric","value"]); w.writerow(["rows",total]); w.writerow(["choice_selected",choice]); w.writerow(["choice_selection_rate",choice/total if total else 0]); w.writerow(["choice_selected_when_not_best",nonbest]); w.writerow(["choice_selected_when_not_best_rate",nonbest/total if total else 0]); w.writerow(["selected_category_is_score_best_rate",best/total if total else 0]); w.writerow(["mean_choice_vs_best_score_gap",sum(gaps)/total if total else 0]); w.writerow([]); w.writerow(["selected_category","count","rate"])
        for c,n in selected.most_common(): w.writerow([c,n,n/total if total else 0])

def run(samples,seed=0,output=None,workers=None):
    if samples<=0: raise ValueError("samples must be positive")
    run_dir=output or Path("logs")/f"choice_selection_{datetime.now().strftime('%m.%d %H_%M')}"; run_dir.mkdir(parents=True,exist_ok=True)
    workers=workers or max(1,(os.cpu_count() or 2)-1); rng=random.Random(seed); payloads=[(i,rng.randrange(2**63)) for i in range(1,samples+1)]
    print(f"Choice Selection: starting {samples:,} first-turn observations ({workers} workers)"); _progress(0,samples,workers); rows=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init) as ex:
        futures=[ex.submit(_observe,p) for p in payloads]
        for done,f in enumerate(as_completed(futures),1): rows.append(f.result()); _progress(done,samples,workers)
    print(); rows.sort(key=lambda r:r["sample"]); stamp=datetime.now().strftime("%m%d%H%M"); data=run_dir/f"choice_selection_{stamp}.csv"; summary=run_dir/f"choice_selection_{stamp}_summary.csv"
    with data.open("w",encoding="utf-8-sig",newline="") as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    _summary(summary,rows); print(f"Saved data: {data}\nSaved summary: {summary}\nTotal first turns written: {len(rows):,}"); return run_dir

def main():
    p=argparse.ArgumentParser(); p.add_argument("--samples",type=int,default=10000); p.add_argument("--seed",type=int,default=0); p.add_argument("--workers",type=int,default=None); p.add_argument("--output",type=Path,default=None); a=p.parse_args(); run(a.samples,a.seed,a.output,a.workers); return 0
if __name__=="__main__": raise SystemExit(main())
