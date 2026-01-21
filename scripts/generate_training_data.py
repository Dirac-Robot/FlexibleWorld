#!/usr/bin/env python3
"""
Goal DSL → Natural Language Augmentation

Goal DSL을 다양한 자연어 명령으로 augment하는 데이터 생성 스크립트.
vLLM 서버를 통해 대규모 자연어 명령 데이터셋을 생성합니다.

Usage:
    # vLLM 서버 시작 후
    python scripts/generate_training_data.py --n-goals 1000 --variations 5
"""
import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


# =============================================================================
# Goal DSL Definition
# =============================================================================

@dataclass
class Condition:
    """조건 표현 (예: >90, <50, =100, 50~100, _)"""
    op: str  # '>', '<', '=', 'range', 'any'
    value: Optional[float] = None
    value2: Optional[float] = None  # for range
    
    def __str__(self):
        if self.op == 'any':
            return '_'
        elif self.op == 'range':
            return f"{self.value}~{self.value2}"
        else:
            return f"{self.op}{self.value}"
    
    def check(self, val: float) -> bool:
        if self.op == 'any':
            return True
        elif self.op == '>':
            return val > self.value
        elif self.op == '<':
            return val < self.value
        elif self.op == '>=':
            return val >= self.value
        elif self.op == '<=':
            return val <= self.value
        elif self.op == '=':
            return abs(val - self.value) < 1.0
        elif self.op == 'range':
            return self.value <= val <= self.value2
        return False
    
    @classmethod
    def random(cls, current_val: float, bounds: tuple = (0, 64)):
        """현재 값 기준으로 랜덤 조건 생성"""
        op = random.choice(['>', '<', '>=', '<=', '=', 'range', 'any'])
        
        if op == 'any':
            return cls(op='any')
        elif op == 'range':
            low = random.uniform(bounds[0], current_val)
            high = random.uniform(current_val, bounds[1])
            return cls(op='range', value=round(low, 1), value2=round(high, 1))
        elif op in ('>', '>='):
            # 현재보다 작은 값으로 조건 설정 (달성 가능하도록)
            target = random.uniform(bounds[0], current_val)
            return cls(op=op, value=round(target, 1))
        elif op in ('<', '<='):
            target = random.uniform(current_val, bounds[1])
            return cls(op=op, value=round(target, 1))
        else:  # =
            return cls(op='=', value=round(current_val + random.uniform(-10, 10), 1))


@dataclass
class GoalDSL:
    """Goal DSL 표현"""
    goal_type: str
    params: dict
    
    def check_success(self, state_before: 'SimState', state_after: 'SimState') -> bool:
        """
        Goal 달성 여부 판정
        Returns: True if goal achieved, False otherwise
        """
        pos_before = np.array(state_before.positions)
        pos_after = np.array(state_after.positions)
        vel_before = np.array(state_before.velocities)
        vel_after = np.array(state_after.velocities)
        
        # 기본 통계량
        center_before = pos_before.mean(axis=0)
        center_after = pos_after.mean(axis=0)
        spread_before = np.std(pos_before)
        spread_after = np.std(pos_after)
        energy_before = np.mean(np.linalg.norm(vel_before, axis=1))
        energy_after = np.mean(np.linalg.norm(vel_after, axis=1))
        
        if self.goal_type == 'move_to':
            pid = self.params['particle_id']
            x_cond = self.params['x_condition']
            y_cond = self.params['y_condition']
            
            p_after = pos_after[pid]
            x_ok = x_cond.check(p_after[0]) if isinstance(x_cond, Condition) else True
            y_ok = y_cond.check(p_after[1]) if isinstance(y_cond, Condition) else True
            return x_ok and y_ok
        
        elif self.goal_type == 'vibrate':
            # 판정: 운동 에너지 증가
            target = self.params.get('target', 'all')
            if target == 'all':
                return energy_after > energy_before * 1.2  # 20% 이상 증가
            else:
                pid = int(target)
                e_before = np.linalg.norm(vel_before[pid])
                e_after = np.linalg.norm(vel_after[pid])
                return e_after > e_before * 1.2
        
        elif self.goal_type == 'cluster':
            # 판정: spread 감소
            return spread_after < spread_before * 0.8  # 20% 이상 감소
        
        elif self.goal_type == 'spread':
            # 판정: spread 증가
            return spread_after > spread_before * 1.2  # 20% 이상 증가
        
        elif self.goal_type == 'move_toward':
            # 판정: 두 입자 사이 거리 감소
            p1, p2 = self.params['particle_id1'], self.params['particle_id2']
            dist_before = np.linalg.norm(pos_before[p1] - pos_before[p2])
            dist_after = np.linalg.norm(pos_after[p1] - pos_after[p2])
            return dist_after < dist_before * 0.8
        
        elif self.goal_type == 'directional_push':
            direction = self.params['direction']
            # 판정: 중심이 해당 방향으로 이동
            if direction == 'up':
                return center_after[1] < center_before[1] - 2  # y 감소 = 위
            elif direction == 'down':
                return center_after[1] > center_before[1] + 2  # y 증가 = 아래
            elif direction == 'left':
                return center_after[0] < center_before[0] - 2  # x 감소 = 왼쪽
            elif direction == 'right':
                return center_after[0] > center_before[0] + 2  # x 증가 = 오른쪽
            return False
        
        elif self.goal_type == 'heat':
            # 판정: 전체 운동 에너지 증가
            return energy_after > energy_before * 1.3
        
        elif self.goal_type == 'cool':
            # 판정: 전체 운동 에너지 감소
            return energy_after < energy_before * 0.7
        
        elif self.goal_type == 'align':
            axis = self.params['axis']
            # 판정: 해당 축의 variance 감소
            if axis == 'horizontal':
                var_before = np.var(pos_before[:, 1])  # y variance
                var_after = np.var(pos_after[:, 1])
                return var_after < var_before * 0.5
            else:  # vertical
                var_before = np.var(pos_before[:, 0])  # x variance
                var_after = np.var(pos_after[:, 0])
                return var_after < var_before * 0.5
        
        return False
    
    def get_success_criteria_text(self) -> str:
        """판정 기준을 텍스트로 반환 (데이터에 포함용)"""
        if self.goal_type == 'move_to':
            x_cond = self.params['x_condition']
            y_cond = self.params['y_condition']
            pid = self.params['particle_id']
            return f"particle[{pid}].x {x_cond} AND particle[{pid}].y {y_cond}"
        elif self.goal_type == 'vibrate':
            target = self.params.get('target', 'all')
            return f"kinetic_energy({target}) increased by 20%+"
        elif self.goal_type == 'cluster':
            return "spread decreased by 20%+"
        elif self.goal_type == 'spread':
            return "spread increased by 20%+"
        elif self.goal_type == 'move_toward':
            p1, p2 = self.params['particle_id1'], self.params['particle_id2']
            return f"distance({p1}, {p2}) decreased by 20%+"
        elif self.goal_type == 'directional_push':
            direction = self.params['direction']
            axis = 'y' if direction in ('up', 'down') else 'x'
            op = '<' if direction in ('up', 'left') else '>'
            return f"center.{axis} {op} before - 2"
        elif self.goal_type == 'heat':
            return "total_kinetic_energy increased by 30%+"
        elif self.goal_type == 'cool':
            return "total_kinetic_energy decreased by 30%+"
        elif self.goal_type == 'align':
            axis = self.params['axis']
            var_axis = 'y' if axis == 'horizontal' else 'x'
            return f"variance({var_axis}) decreased by 50%+"
        return "unknown"
    
    def to_dsl_string(self) -> str:
        """DSL 문자열로 변환"""
        if self.goal_type == 'move_to':
            pid = self.params['particle_id']
            x_cond = self.params['x_condition']
            y_cond = self.params['y_condition']
            return f"move particle {pid} to ({x_cond}, {y_cond})"
        
        elif self.goal_type == 'vibrate':
            target = self.params.get('target', self.params.get('particle_id', 'all'))
            intensity = self.params['intensity']
            return f"make particle {target} vibrate {intensity}"
        
        elif self.goal_type == 'cluster':
            pids = self.params['particle_ids']
            return f"cluster particles {pids}"
        
        elif self.goal_type == 'spread':
            pids = self.params['particle_ids']
            return f"spread particles {pids}"
        
        elif self.goal_type == 'move_toward':
            p1 = self.params['particle_id1']
            p2 = self.params['particle_id2']
            return f"move particle {p1} toward particle {p2}"
        
        elif self.goal_type == 'align':
            axis = self.params['axis']
            pids = self.params.get('particle_ids', 'all')
            return f"align particles {pids} {axis}ly"
        
        elif self.goal_type == 'heat':
            target = self.params.get('target', 'all')
            amount = self.params.get('amount', 'moderate')
            return f"apply {amount} heat to {target}"
        
        elif self.goal_type == 'cool':
            target = self.params.get('target', 'all')
            return f"cool down {target}"
        
        elif self.goal_type == 'directional_push':
            direction = self.params['direction']
            return f"push all particles {direction}"
        
        else:
            return f"{self.goal_type}: {self.params}"
    
    def get_semantic_hints(self) -> list:
        """LLM에게 줄 의미적 힌트"""
        hints = []
        
        if self.goal_type == 'move_to':
            x_cond = self.params['x_condition']
            y_cond = self.params['y_condition']
            
            # X 방향 힌트
            if isinstance(x_cond, Condition):
                if x_cond.op == '>':
                    hints.append("오른쪽으로 이동")
                elif x_cond.op == '<':
                    hints.append("왼쪽으로 이동")
            
            # Y 방향 힌트
            if isinstance(y_cond, Condition):
                if y_cond.op == '>':
                    hints.append("아래로 이동")
                elif y_cond.op == '<':
                    hints.append("위로 이동")
        
        elif self.goal_type == 'vibrate':
            intensity = self.params.get('intensity', 'moderate')
            if intensity in ('high', 'strong', 'intense'):
                hints.append("강하게 진동/열")
            else:
                hints.append("약하게 진동/열")
        
        elif self.goal_type == 'cluster':
            hints.append("모으기/뭉치기/집중")
        
        elif self.goal_type == 'spread':
            hints.append("퍼뜨리기/분산/폭발")
        
        elif self.goal_type == 'directional_push':
            direction = self.params['direction']
            hints.append(f"{direction} 방향으로 밀기")
        
        return hints


@dataclass 
class SimState:
    """시뮬레이션 상태"""
    n_particles: int
    positions: list  # [[x, y], ...]
    velocities: list  # [[vx, vy], ...]
    
    def summary(self) -> str:
        """상태 요약 문자열"""
        pos = np.array(self.positions)
        center = pos.mean(axis=0)
        spread = np.std(pos)
        return f"{self.n_particles} particles, center at ({center[0]:.0f}, {center[1]:.0f}), spread: {spread:.1f}"
    
    @classmethod
    def random(cls, n_particles: int = 8, bounds: tuple = (10, 54)):
        positions = np.random.uniform(bounds[0], bounds[1], (n_particles, 2)).tolist()
        velocities = np.random.uniform(-1, 1, (n_particles, 2)).tolist()
        return cls(n_particles=n_particles, positions=positions, velocities=velocities)


# =============================================================================
# Goal Generators
# =============================================================================

def generate_move_to_goal(state: SimState) -> GoalDSL:
    """move_to 목표 생성"""
    pid = random.randint(0, state.n_particles - 1)
    current_pos = state.positions[pid]
    
    x_cond = Condition.random(current_pos[0])
    y_cond = Condition.random(current_pos[1])
    
    return GoalDSL(
        goal_type='move_to',
        params={
            'particle_id': pid,
            'x_condition': x_cond,
            'y_condition': y_cond,
            'current_pos': current_pos,
        }
    )


def generate_vibrate_goal(state: SimState) -> GoalDSL:
    """vibrate 목표 생성"""
    target = random.choice([random.randint(0, state.n_particles - 1), 'all'])
    intensity = random.choice(['low', 'moderate', 'high', 'intense'])
    
    return GoalDSL(
        goal_type='vibrate',
        params={
            'target': target,
            'intensity': intensity,
        }
    )


def generate_cluster_goal(state: SimState) -> GoalDSL:
    """cluster 목표 생성"""
    n_select = random.randint(2, state.n_particles)
    pids = random.sample(range(state.n_particles), n_select)
    
    return GoalDSL(
        goal_type='cluster',
        params={
            'particle_ids': pids if len(pids) < state.n_particles else 'all',
        }
    )


def generate_spread_goal(state: SimState) -> GoalDSL:
    """spread 목표 생성"""
    n_select = random.randint(2, state.n_particles)
    pids = random.sample(range(state.n_particles), n_select)
    
    return GoalDSL(
        goal_type='spread',
        params={
            'particle_ids': pids if len(pids) < state.n_particles else 'all',
        }
    )


def generate_move_toward_goal(state: SimState) -> GoalDSL:
    """move_toward 목표 생성"""
    p1, p2 = random.sample(range(state.n_particles), 2)
    
    return GoalDSL(
        goal_type='move_toward',
        params={
            'particle_id1': p1,
            'particle_id2': p2,
        }
    )


def generate_directional_push_goal(state: SimState) -> GoalDSL:
    """directional_push 목표 생성"""
    direction = random.choice(['up', 'down', 'left', 'right'])
    
    return GoalDSL(
        goal_type='directional_push',
        params={
            'direction': direction,
        }
    )


def generate_heat_goal(state: SimState) -> GoalDSL:
    """heat 목표 생성"""
    target = random.choice(['all', random.randint(0, state.n_particles - 1)])
    amount = random.choice(['slight', 'moderate', 'strong', 'extreme'])
    
    return GoalDSL(
        goal_type='heat',
        params={
            'target': target,
            'amount': amount,
        }
    )


def generate_align_goal(state: SimState) -> GoalDSL:
    """align 목표 생성"""
    axis = random.choice(['horizontal', 'vertical'])
    
    return GoalDSL(
        goal_type='align',
        params={
            'axis': axis,
            'particle_ids': 'all',
        }
    )


GOAL_GENERATORS = {
    'move_to': generate_move_to_goal,
    'vibrate': generate_vibrate_goal,
    'cluster': generate_cluster_goal,
    'spread': generate_spread_goal,
    'move_toward': generate_move_toward_goal,
    'directional_push': generate_directional_push_goal,
    'heat': generate_heat_goal,
    'align': generate_align_goal,
}

GOAL_WEIGHTS = {
    'move_to': 3.0,  # 가장 빈번
    'vibrate': 2.0,
    'cluster': 2.0,
    'spread': 2.0,
    'move_toward': 1.5,
    'directional_push': 2.0,
    'heat': 1.5,
    'align': 1.0,
}


def sample_goal(state: SimState) -> GoalDSL:
    """가중치에 따라 랜덤 goal 샘플링"""
    goal_types = list(GOAL_GENERATORS.keys())
    weights = [GOAL_WEIGHTS[g] for g in goal_types]
    total = sum(weights)
    weights = [w/total for w in weights]
    
    goal_type = np.random.choice(goal_types, p=weights)
    return GOAL_GENERATORS[goal_type](state)


# =============================================================================
# LLM Augmentation
# =============================================================================

AUGMENTATION_PROMPT_TEMPLATE = """당신은 로봇 제어 명령 데이터를 생성하는 전문가입니다.

## Task
주어진 Goal DSL을 다양한 자연어 명령으로 변환하세요. 
같은 의미를 가지지만 다양한 표현, 구체성 수준, 언어(한국어/영어)로 변환합니다.

## Goal DSL 문법
- `move particle N to (x_cond, y_cond)`: 입자 이동
  - 조건: `>N`, `<N`, `=N`, `N~M` (범위), `_` (상관없음)
  - `>90` = x좌표가 90보다 크게 = 오른쪽으로
  - `<30` = x좌표가 30보다 작게 = 왼쪽으로
  - y좌표: `>N` = 아래로, `<N` = 위로 (y축이 아래로 증가)

- `make particle N vibrate INTENSITY`: 입자 진동
  - 열을 가하거나 에너지를 주는 것과 같음

- `cluster particles [ids]`: 입자들 모으기
  - 인력, 모으기, 뭉치기

- `spread particles [ids]`: 입자들 퍼뜨리기  
  - 척력, 폭발, 분산

- `push all particles DIRECTION`: 방향으로 밀기

- `apply heat to TARGET`: 열 가하기

## Rules
1. 다양한 구체성 수준:
   - 정확한: "입자 1을 x=95 위치로 이동시켜"
   - 범용적: "입자를 오른쪽으로 움직여"
   - 추상적: "오른쪽으로 밀어"

2. 다양한 언어:
   - 한국어 (70%)
   - 영어 (30%)

3. 다양한 어조:
   - 명령형: "움직여", "Move it"
   - 요청형: "움직여 주세요", "Please move"
   - 설명형: "오른쪽으로 가야 해"

4. 물리적 표현 변환:
   - vibrate = 진동 = 열 = 에너지
   - cluster = 모으기 = 인력 = 응집
   - spread = 퍼뜨리기 = 척력 = 폭발

## Current Goal
DSL: {dsl_string}
Semantic hints: {hints}
State: {state_summary}

## Output Format
정확히 {n_variations}개의 자연어 명령을 생성하세요.
각 명령은 한 줄씩, 번호 없이 출력합니다.

예시 출력:
입자를 오른쪽으로 움직여
Move the particle to the right
입자 1을 x=95로 이동시켜
오른쪽으로 밀어줘
첫 번째 입자를 오른쪽으로 조금 이동해"""


@dataclass
class AugmentedSample:
    """Augmented 데이터 샘플"""
    goal_dsl: str
    goal_type: str
    goal_params: dict
    success_criteria: str  # 판정 기준
    state: dict
    natural_commands: list
    hints: list


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "Qwen/Qwen2.5-32B-Instruct"):
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = model
    
    def augment(self, goal: GoalDSL, state: SimState, n_variations: int = 5) -> list:
        """Goal DSL을 자연어 명령들로 augment"""
        prompt = AUGMENTATION_PROMPT_TEMPLATE.format(
            dsl_string=goal.to_dsl_string(),
            hints=goal.get_semantic_hints(),
            state_summary=state.summary(),
            n_variations=n_variations,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=500,
            )
            
            content = response.choices[0].message.content.strip()
            commands = [line.strip() for line in content.split('\n') if line.strip()]
            # 번호 제거 (1. 2. 등)
            commands = [c.lstrip('0123456789.-) ').strip() for c in commands]
            commands = [c for c in commands if len(c) > 2]
            
            return commands[:n_variations]
        
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return []


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_single_sample(client: VLLMClient, n_variations: int = 5) -> Optional[AugmentedSample]:
    """단일 샘플 생성"""
    state = SimState.random()
    goal = sample_goal(state)
    
    commands = client.augment(goal, state, n_variations)
    
    if not commands:
        return None
    
    # Condition 객체를 직렬화 가능하게 변환
    params = {}
    for k, v in goal.params.items():
        if isinstance(v, Condition):
            params[k] = str(v)
        elif isinstance(v, np.ndarray):
            params[k] = v.tolist()
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.floating):
            params[k] = [float(x) for x in v]
        else:
            params[k] = v
    
    return AugmentedSample(
        goal_dsl=goal.to_dsl_string(),
        goal_type=goal.goal_type,
        goal_params=params,
        success_criteria=goal.get_success_criteria_text(),
        state={
            'n_particles': state.n_particles,
            'positions': [[float(x) for x in p] for p in state.positions],
            'velocities': [[float(x) for x in v] for v in state.velocities],
        },
        natural_commands=commands,
        hints=goal.get_semantic_hints(),
    )


def generate_dataset(
    client: VLLMClient,
    n_goals: int,
    n_variations: int,
    output_path: Path,
    n_workers: int = 8,
):
    """데이터셋 생성"""
    samples = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(generate_single_sample, client, n_variations)
            for _ in range(n_goals)
        ]
        
        for future in tqdm(as_completed(futures), total=n_goals, desc="Generating"):
            try:
                sample = future.result()
                if sample:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Sample generation failed: {e}")
    
    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Training format (각 command가 별도 샘플)
    with open(output_path, 'w') as f:
        for sample in samples:
            for cmd in sample.natural_commands:
                entry = {
                    "goal_dsl": sample.goal_dsl,
                    "goal_type": sample.goal_type,
                    "success_criteria": sample.success_criteria,
                    "natural_command": cmd,
                    "state": sample.state,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 2. Full format (분석용)
    full_path = output_path.with_suffix('.full.json')
    with open(full_path, 'w') as f:
        json.dump([asdict(s) for s in samples], f, indent=2, ensure_ascii=False)
    
    # Statistics
    total_commands = sum(len(s.natural_commands) for s in samples)
    logger.info(f"Generated {len(samples)} goals → {total_commands} commands")
    logger.info(f"Saved to {output_path} and {full_path}")
    
    # Goal type distribution
    from collections import Counter
    goal_types = [s.goal_type for s in samples]
    logger.info(f"Goal distribution: {Counter(goal_types)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Goal DSL → Natural Language data")
    parser.add_argument("--n-goals", type=int, default=1000, help="Number of goals to generate")
    parser.add_argument("--n-variations", type=int, default=5, help="Natural language variations per goal")
    parser.add_argument("--output", type=str, default="/workspace/Projects/FlexibleWorld/data/goal_commands.jsonl")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--n-workers", type=int, default=8)
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Goal DSL → Natural Language Augmentation")
    logger.info("=" * 50)
    logger.info(f"Goals: {args.n_goals}")
    logger.info(f"Variations per goal: {args.n_variations}")
    logger.info(f"Total commands: ~{args.n_goals * args.n_variations}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 50)
    
    client = VLLMClient(base_url=args.vllm_url, model=args.model)
    
    generate_dataset(
        client=client,
        n_goals=args.n_goals,
        n_variations=args.n_variations,
        output_path=Path(args.output),
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
