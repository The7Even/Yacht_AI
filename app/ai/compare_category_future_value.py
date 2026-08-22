"""Compare Choice against 4K / Full House with bounded future rollouts."""
from __future__ import annotations
import argparse,csv,os,random
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from app.core.categories import Category
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine,MAX_ROLLS_PER_TURN
from app.core.game_state import PlayerId
from app.core.scoring import ScoreCalculator
from .action_generator import ActionType
from .expected_value_strategy import ExpectedValueStrategy
FIELDS=("sample","rollout","category","dice","choice_score","target_score","score_gap","choice_horizon_score","target_horizon_score","delta_target_minus_choice","choice_bonus","target_bonus")
_WORKER_STRATEGY=None

def _worker_init():
 global _WORKER_STRATEGY; _WORKER_STRATEGY=ExpectedValueStrategy()

def _strategy():
 global _WORKER_STRATEGY
 if _WORKER_STRATEGY is None: _WORKER_STRATEGY=ExpectedValueStrategy()
 return _WORKER_STRATEGY

def _candidate_hands(category):
 hands=combinations_with_replacement(range(1,7),5)
 if category is Category.FOUR_OF_A_KIND:
  return tuple(h for h in hands if max(h.count(f) for f in set(h))==4 and ScoreCalculator.calculate(category,h)>0)
 if category is Category.FULL_HOUSE:
  return tuple(h for h in hands if sorted(h.count(f) for f in set(h))==[2,3])
 raise ValueError("Only FOUR_OF_A_KIND and FULL_HOUSE are supported.")
_HANDS={Category.FOUR_OF_A_KIND:_candidate_hands(Category.FOUR_OF_A_KIND),Category.FULL_HOUSE:_candidate_hands(Category.FULL_HOUSE)}

def _play_future(forced_category,initial_dice,seed,horizon):
 engine=GameEngine(dice_roller=DiceRoller(random.Random(seed))); engine.start_game(); engine.state.current_dice=initial_dice; engine.state.roll_count=MAX_ROLLS_PER_TURN; engine.score_category(forced_category); strategy=_strategy(); turns=0
 while not engine.state.game_over and turns<horizon:
  engine.state.current_player=PlayerId.PLAYER; engine.state.current_dice=None; engine.state.held_indices=frozenset(); engine.state.roll_count=0; engine.state.turn_scored=False
  while not engine.state.turn_scored:
   engine.roll_dice(); action=strategy.decide(engine.state).action
   if action.type is ActionType.SCORE:
    assert action.selected_category is not None; engine.score_category(action.selected_category); turns+=1; break
   desired=frozenset(action.held_indices); current=engine.state.held_indices
   for i in sorted(current-desired): engine.unhold_dice(i)
   for i in sorted(desired-current): engine.hold_dice(i)
   if engine.state.roll_count>=MAX_ROLLS_PER_TURN:
    action=strategy.decide(engine.state).action; assert action.selected_category is not None; engine.score_category(action.selected_category); turns+=1; break
 p=engine.state.players[PlayerId.PLAYER]; return p.total_score,p.has_upper_bonus

def _observe(payload):
 sample,rollout,category,dice,seed,horizon=payload; choice=ScoreCalculator.calculate(Category.CHOICE,dice); target=ScoreCalculator.calculate(category,dice)
 cf,cb=_play_future(Category.CHOICE,dice,seed,horizon); tf,tb=_play_future(category,dice,seed,horizon)
 return {"sample":sample,"rollout":rollout,"category":category.value,"dice":"".join(map(str,dice)),"choice_score":choice,"target_score":target,"score_gap":target-choice,"choice_horizon_score":cf,"target_horizon_score":tf,"delta_target_minus_choice":tf-cf,"choice_bonus":int(cb),"target_bonus":int(tb)}

def _progress(done,total,workers):
 pct=done/total*100 if total else 100; width=40; filled=int(width*pct/100); print(f"\r4K/FH Future Value [{'#'*filled+'.'*(width-filled)}] {pct:6.2f}% | {done}/{total} branches | {workers} workers",end="",flush=True)

def run(samples,rollouts,category,horizon=5,seed=0,workers=None,output=None):
 if samples<=0 or rollouts<=0 or horizon<=0: raise ValueError("samples, rollouts, and horizon must be positive")
 run_dir=output or Path("logs")/f"category_future_value_{datetime.now().strftime('%m.%d %H_%M')}"; run_dir.mkdir(parents=True,exist_ok=True); workers=workers or max(1,(os.cpu_count() or 2)-1); rng=random.Random(seed); hands=_HANDS[category]
 payloads=[]
 for sample in range(1,samples+1):
  dice=hands[rng.randrange(len(hands))]
  for rollout in range(1,rollouts+1): payloads.append((sample,rollout,category,dice,rng.randrange(2**63),horizon))
 total=len(payloads); print(f"Category Future Value: {category.value} | {samples:,} hands × {rollouts:,} rollouts × 2 branches | {horizon} future turns"); _progress(0,total,workers); rows=[]
 with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init) as ex:
  futures=[ex.submit(_observe,p) for p in payloads]
  for done,f in enumerate(as_completed(futures),1): rows.append(f.result()); _progress(done,total,workers)
 print(); rows.sort(key=lambda r:(int(r['sample']),int(r['rollout']))); stamp=datetime.now().strftime('%m%d%H%M'); data=run_dir/f'category_future_value_{category.value}_{stamp}.csv'; summary=run_dir/f'category_future_value_{category.value}_{stamp}_summary.csv'
 with data.open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
 deltas=[float(r['delta_target_minus_choice']) for r in rows]; gaps=[int(r['score_gap']) for r in rows]; tw=sum(d>0 for d in deltas); cw=sum(d<0 for d in deltas); draws=len(deltas)-tw-cw
 with summary.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f); w.writerow(['metric','value']); w.writerow(['category',category.value]); w.writerow(['hands',samples]); w.writerow(['rollouts_per_hand',rollouts]); w.writerow(['branches_per_category',len(rows)]); w.writerow(['future_turn_horizon',horizon]); w.writerow(['mean_score_gap_target_minus_choice',sum(gaps)/len(gaps)]); w.writerow(['mean_horizon_delta_target_minus_choice',sum(deltas)/len(deltas)]); w.writerow(['target_wins',tw]); w.writerow(['target_win_rate',tw/len(deltas)]); w.writerow(['choice_wins',cw]); w.writerow(['choice_win_rate',cw/len(deltas)]); w.writerow(['draws',draws]); w.writerow(['draw_rate',draws/len(deltas)]); w.writerow(['target_bonus_rate',sum(int(r['target_bonus']) for r in rows)/len(rows)]); w.writerow(['choice_bonus_rate',sum(int(r['choice_bonus']) for r in rows)/len(rows)])
 print(f"Saved data: {data}\nSaved summary: {summary}"); return run_dir

def main():
 p=argparse.ArgumentParser(description='Compare Choice vs 4K/Full House future value with bounded rollouts.'); p.add_argument('--category',choices=(Category.FOUR_OF_A_KIND.value,Category.FULL_HOUSE.value),required=True); p.add_argument('--samples',type=int,default=1000); p.add_argument('--rollouts',type=int,default=10); p.add_argument('--horizon',type=int,default=5); p.add_argument('--seed',type=int,default=0); p.add_argument('--workers',type=int,default=None); p.add_argument('--output',type=Path,default=None); a=p.parse_args(); run(a.samples,a.rollouts,Category(a.category),a.horizon,a.seed,a.workers,a.output); return 0
if __name__=='__main__': raise SystemExit(main())
