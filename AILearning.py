import collections
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CATEGORIES = [
    "Aces",
    "Deuces",
    "Threes",
    "Fours",
    "Fives",
    "Sixes",
    "Choice",
    "4 of a Kind",
    "Full House",
    "S. Straight",
    "L. Straight",
    "Yacht",
]


def calculate_category_score(cat_idx, dice):
  counts = collections.Counter(dice)
  total = sum(dice)
  if 0 <= cat_idx <= 5:
    target_num = cat_idx + 1
    return dice.count(target_num) * target_num
  elif cat_idx == 6:
    return total
  elif cat_idx == 7:
    return total if any(cnt >= 4 for cnt in counts.values()) else 0
  elif cat_idx == 8:
    vals = sorted(counts.values())
    return total if (vals == [2, 3] or vals == [5]) else 0
  elif cat_idx == 9:
    unique_dice = set(dice)
    straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
    return 15 if any(s.issubset(unique_dice) for s in straights) else 0
  elif cat_idx == 10:
    unique_dice = sorted(list(set(dice)))
    return (
        30
        if (unique_dice == [1, 2, 3, 4, 5] or unique_dice == [2, 3, 4, 5, 6])
        else 0
    )
  elif cat_idx == 11:
    return 50 if len(counts) == 1 else 0
  return 0


# ==========================================
# 상단 보너스 집중 공략 스마트 리롤
# ==========================================
def smart_reroll_turn(used_mask, total_upper):
  dice = [random.randint(1, 6) for _ in range(5)]

  for _ in range(2):
    counts = collections.Counter(dice)
    most_num, most_cnt = counts.most_common(1)[0]
    unique_dice = sorted(list(set(dice)))

    # 1. 요트 기회 (요트 비어있고 4개 이상 일치 시 최우선)
    if not used_mask[11] and most_cnt >= 4:
      keep = [d for d in dice if d == most_num]
      dice = keep + [random.randint(1, 6) for _ in range(5 - len(keep))]
      continue

    # 2. [핵심] 상단 63점 미달성 시 공격적 킵
    if total_upper < 63:
      # 63점까지 얼마 안 남았을 때는 1개만 있어도 상단 빈칸 노림
      min_req_count = 1 if total_upper >= 45 else 2
      upper_candidates = [
          num
          for num in [6, 5, 4, 3, 2, 1]
          if not used_mask[num - 1] and counts.get(num, 0) >= min_req_count
      ]
      if upper_candidates:
        target_num = upper_candidates[0]
        keep = [d for d in dice if d == target_num]
        dice = keep + [random.randint(1, 6) for _ in range(5 - len(keep))]
        continue

    # 3. 라지 / 스몰 스트레이트 노림수
    if not used_mask[10] or not used_mask[9]:
      straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
      matched = [s for s in straights if len(s.intersection(unique_dice)) >= 4]
      if matched:
        keep = list(matched[0].intersection(unique_dice))
        dice = keep + [random.randint(1, 6) for _ in range(5 - len(keep))]
        continue

    # 4. 풀하우스 노림수
    vals = sorted(counts.values())
    if not used_mask[8] and vals == [1, 2, 2]:
      keep = [d for d in dice if counts[d] == 2]
      dice = keep + [random.randint(1, 6)]
      continue

    # 5. 기본: 최빈값 킵
    keep = [d for d in dice if d == most_num]
    dice = keep + [random.randint(1, 6) for _ in range(5 - len(keep))]

  return dice


# ==========================================
# DQN 네트워크 (30차원 입력)
# ==========================================
class YachtDQN(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(30, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 12),
    )

  def forward(self, x):
    return self.net(x)


def make_state_vector(dice, used_mask, scores, total_upper):
  dice_norm = [d / 6.0 for d in sorted(dice)]
  mask_norm = [float(m) for m in used_mask]
  scores_norm = [s / 50.0 for s in scores]
  upper_progress = [min(1.0, total_upper / 63.0)]
  return np.array(
      dice_norm + mask_norm + scores_norm + upper_progress, dtype=np.float32
  )


# ==========================================
# 20,000판 보너스 특화 파이프라인
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YachtDQN().to(device)
target_model = copy.deepcopy(model).to(device)
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.SmoothL1Loss()

total_games = 20000
batch_interval = 1000
epsilon = 0.95
memory = collections.deque(maxlen=30000)
batch_size = 64

history_total_scores = []
history_upper_scores = []
history_bonus_flags = []
global_max_score = 0

print(f"학습 시작 디바이스: {device}")
print(f"평균 150점 목표 20,000게임 보너스 특화 가동 중...")

for game in range(1, total_games + 1):
  used_mask = [False] * 12
  scores = [0] * 12
  total_upper = 0

  current_dice = smart_reroll_turn(used_mask, total_upper)

  for turn in range(12):
    state_vec = make_state_vector(current_dice, used_mask, scores, total_upper)
    state_t = torch.tensor(state_vec).unsqueeze(0).to(device)
    avail_indices = [i for i, u in enumerate(used_mask) if not u]

    if random.random() < epsilon:
      chosen_cat = random.choice(avail_indices)
    else:
      with torch.no_grad():
        q_vals = model(state_t).squeeze(0).cpu().numpy()
        q_vals[[i for i in range(12) if used_mask[i]]] = -999999.0
        chosen_cat = int(np.argmax(q_vals))

    gain = calculate_category_score(chosen_cat, current_dice)
    scores[chosen_cat] = gain
    used_mask[chosen_cat] = True

    # 보상 설계 (상단 보너스 최우선 유도)
    reward = float(gain)

    # 1. 0점 희생 페널티 세분화
    if gain == 0 and chosen_cat in [10, 11]:
      reward -= 20.0  # 요트, L.스트레이트 자폭 방지
    elif gain == 0 and chosen_cat in [0, 1]:
      reward -= 2.0  # Aces, Deuces는 안전한 버림패로 인정

    # 2. 상단 보너스 강력 유도
    if chosen_cat <= 5:
      total_upper += gain
      target_std = (chosen_cat + 1) * 3
      if gain >= target_std:
        reward += 16.0  # 정량(3개) 이상 획득 시 보너스 리워드
      elif gain == 0:
        reward -= 8.0

      # 63점 돌파 순간 메가 보너스 리워드
      if total_upper >= 63 and (total_upper - gain) < 63:
        reward += 80.0

    next_dice = (
        smart_reroll_turn(used_mask, total_upper) if turn < 11 else [0] * 5
    )
    next_state_vec = make_state_vector(
        next_dice, used_mask, scores, total_upper
    )

    memory.append((
        state_vec,
        chosen_cat,
        reward,
        next_state_vec,
        list(used_mask),
        turn == 11,
    ))
    current_dice = next_dice

    if len(memory) >= batch_size:
      batch = random.sample(memory, batch_size)
      b_s, b_a, b_r, b_ns, b_m, b_done = zip(*batch)

      b_s_t = torch.tensor(np.array(b_s)).to(device)
      b_ns_t = torch.tensor(np.array(b_ns)).to(device)
      b_a_t = torch.tensor(b_a).unsqueeze(1).to(device)
      b_r_t = torch.tensor(b_r, dtype=torch.float32).unsqueeze(1).to(device)
      b_done_t = (
          torch.tensor(b_done, dtype=torch.float32).unsqueeze(1).to(device)
      )

      curr_q = model(b_s_t).gather(1, b_a_t)

      with torch.no_grad():
        next_q_all = target_model(b_ns_t)
        for row_i, m in enumerate(b_m):
          u_idx = [idx for idx, u in enumerate(m) if u]
          if len(u_idx) < 12:
            next_q_all[row_i, u_idx] = -999999.0
        max_next_q = next_q_all.max(1)[0].unsqueeze(1)
        target_q = b_r_t + (1.0 - b_done_t) * 0.95 * max_next_q

      loss = criterion(curr_q, target_q)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  epsilon = max(0.015, epsilon * 0.99975)

  if game % 250 == 0:
    target_model.load_state_dict(model.state_dict())

  has_bonus = total_upper >= 63
  final_score = sum(scores) + (35 if has_bonus else 0)
  history_total_scores.append(final_score)
  history_upper_scores.append(total_upper)
  history_bonus_flags.append(1 if has_bonus else 0)

  if final_score > global_max_score:
    global_max_score = final_score

  if game % batch_interval == 0:
    avg_tot = np.mean(history_total_scores[-batch_interval:])
    avg_up = np.mean(history_upper_scores[-batch_interval:])
    b_rate = (sum(history_bonus_flags[-batch_interval:]) / batch_interval) * 100
    recent_max = np.max(history_total_scores[-batch_interval:])

    print(
        f"[{game:>5}/{total_games}] 최근 {batch_interval}판 -> 평균:"
        f" {avg_tot:.1f}점 | 구간최고: {recent_max}점 (역대최고:"
        f" {global_max_score}점) | 상단합: {avg_up:.1f}점 | 보너스:"
        f" {b_rate:.1f}% | eps: {epsilon:.3f}"
    )

save_path = "yacht_ai_brain.pth"
torch.save(model.state_dict(), save_path)
print(f"\n학습 완료! 보너스 특화 최강 두뇌 저장 완료: {save_path}")
