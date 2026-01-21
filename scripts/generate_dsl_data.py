#!/usr/bin/env python3
"""
DSL-based Training Data Generation using vLLM

시뮬레이터 상태를 LLM에 제공하고, DSL 명령어를 생성하게 함.
생성된 DSL을 실행하고 결과를 저장.

Usage:
    # vLLM 서버 시작 후
    python scripts/generate_dsl_data.py --n-samples 1000 --output data/dsl_training.jsonl
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.core import ParticleSimulator
from simulator.action_operator import ActionOperator, ActionType
from simulator.dsl import MetaDSL, create_dsl, DSLParseError
from simulator.meta_action import create_registry


# =============================================================================
# State Description
# =============================================================================

@dataclass
class SimState:
    """Simulation state snapshot"""
    n_particles: int
    positions: List[List[float]]
    velocities: List[List[float]]
    mean_pos: List[float]
    spread: float
    mean_velocity: float

    @classmethod
    def from_simulator(cls, sim: ParticleSimulator) -> 'SimState':
        active = sim.active
        n = int(active.sum())
        
        if n == 0:
            return cls(0, [], [], [32, 32], 0, 0)
        
        positions = sim.positions[active][:n].tolist()
        velocities = sim.velocities[active][:n].tolist()
        
        pos_array = np.array(positions)
        vel_array = np.array(velocities)
        
        mean_pos = pos_array.mean(axis=0).tolist()
        spread = float(pos_array.std())
        mean_velocity = float(np.linalg.norm(vel_array, axis=1).mean())
        
        return cls(n, positions, velocities, mean_pos, spread, mean_velocity)

    def describe(self) -> str:
        """Generate natural language description of state"""
        if self.n_particles == 0:
            return "The simulation is empty with no particles."
        
        desc = f"There are {self.n_particles} particles in the simulation. "
        
        # Position description
        cx, cy = self.mean_pos
        if cx < 24:
            h_pos = "left side"
        elif cx > 40:
            h_pos = "right side"
        else:
            h_pos = "center"
            
        if cy < 24:
            v_pos = "top"
        elif cy > 40:
            v_pos = "bottom"
        else:
            v_pos = "middle"
        
        desc += f"The particles are concentrated in the {v_pos} {h_pos} of the arena. "
        
        # Spread description
        if self.spread < 5:
            desc += "They are tightly clustered together. "
        elif self.spread < 15:
            desc += "They are moderately spread out. "
        else:
            desc += "They are widely dispersed across the area. "
        
        # Velocity description
        if self.mean_velocity < 0.5:
            desc += "The particles are mostly stationary."
        elif self.mean_velocity < 2:
            desc += "The particles are moving slowly."
        else:
            desc += "The particles are moving rapidly."
        
        return desc


# =============================================================================
# Goal Templates
# =============================================================================

GOALS = [
    # Structure creation
    "Create a rigid body at the center with 6 particles",
    "Create a membrane around the particles",
    "Create a chain connecting the left and right sides",
    "Create a soft body in the upper left corner",
    "Create a small rigid triangle at position (20, 20)",
    "Create a large membrane with 20 particles",
    "Create a polymer chain from (10, 32) to (54, 32)",
    
    # Force application
    "Push all particles to the left",
    "Push all particles upward",
    "Apply attraction at the center to gather particles",
    "Apply repulsion to spread the particles apart",
    "Heat up the center area to increase particle movement",
    "Apply a strong force at (40, 40) pushing right",
    
    # Complex goals
    "Create two rigid bodies and push them together",
    "Create a membrane and then heat the inside",
    "Build a cell-like structure with membrane and internal particles",
    "Create a chain and apply force to make it swing",
    
    # Multi-step
    "First create particles, then cluster them together",
    "Create a structure and apply forces to test its stability",
]


# =============================================================================
# LLM DSL Generator
# =============================================================================

DSL_SYSTEM_PROMPT = """You are a physics simulation controller. You generate DSL commands to control a 2D particle simulation.

Available DSL commands:
1. CREATE <type> AT (x, y) WITH <params>
   Types: rigid_body, soft_body, membrane, chain, particle
   Params: n=<num>, radius=<r>, shape=<circle|square|triangle|line>
   
2. CREATE chain FROM (x1, y1) TO (x2, y2) WITH n=<num>

3. APPLY <force> AT (x, y) WITH value=<v> radius=<r>
   Forces: force, attraction, repulsion, heat

4. STEP - advance simulation by one step
5. WAIT <n> - advance simulation by n steps

The simulation area is 64x64 pixels. Center is (32, 32).

Examples:
- "CREATE rigid_body AT (32, 32) WITH n=6 shape=circle radius=10"
- "CREATE membrane AT (32, 32) WITH n=16 outer=12"
- "CREATE chain FROM (10, 32) TO (54, 32) WITH n=8"
- "APPLY attraction AT (32, 32) WITH value=3 radius=25"
- "APPLY force AT (20, 32) WITH value=5 radius=20"

Generate only DSL commands, one per line. No explanations."""


def create_llm_client(base_url: str = "http://localhost:8000/v1") -> OpenAI:
    """Create OpenAI client for vLLM server"""
    return OpenAI(base_url=base_url, api_key="dummy")


def generate_dsl_from_llm(
    client: OpenAI,
    model: str,
    state_desc: str,
    goal: str,
    temperature: float = 0.7,
) -> str:
    """Generate DSL commands from LLM"""
    
    user_prompt = f"""Current state:
{state_desc}

Goal: {goal}

Generate DSL commands to achieve this goal:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DSL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=512,
    )
    
    return response.choices[0].message.content.strip()


# =============================================================================
# Data Generation Pipeline
# =============================================================================

@dataclass
class DSLSample:
    """Training sample for DSL generation"""
    state_description: str
    goal: str
    dsl_commands: str
    parsed_successfully: bool
    n_actions: int
    state_before: Dict[str, Any]
    state_after: Optional[Dict[str, Any]] = None


def create_random_initial_state(sim: ParticleSimulator, n_particles: int = None):
    """Create random initial state"""
    sim.reset()
    
    if n_particles is None:
        n_particles = random.randint(5, 30)
    
    for _ in range(n_particles):
        x = random.uniform(10, 54)
        y = random.uniform(10, 54)
        sim.add_particle(x, y)
    
    # Run a few steps to settle
    for _ in range(5):
        sim.update()


def generate_sample(
    client: OpenAI,
    model: str,
    dsl: MetaDSL,
    sim: ParticleSimulator,
    goal: str,
) -> Optional[DSLSample]:
    """Generate a single training sample"""
    
    try:
        # Create initial state
        create_random_initial_state(sim)
        state_before = SimState.from_simulator(sim)
        state_desc = state_before.describe()
        
        # Generate DSL from LLM
        dsl_output = generate_dsl_from_llm(client, model, state_desc, goal)
        
        # Try to parse
        try:
            actions = dsl.parse_multi(dsl_output)
            parsed_ok = True
            n_actions = len(actions)
        except DSLParseError:
            parsed_ok = False
            n_actions = 0
            actions = []
        
        # Execute if parsed
        state_after = None
        if parsed_ok and actions:
            operator = ActionOperator(sim)
            for action in actions:
                try:
                    operator.execute(action)
                except Exception:
                    pass
            
            # Run simulation
            for _ in range(20):
                sim.update()
            
            state_after = asdict(SimState.from_simulator(sim))
        
        return DSLSample(
            state_description=state_desc,
            goal=goal,
            dsl_commands=dsl_output,
            parsed_successfully=parsed_ok,
            n_actions=n_actions,
            state_before=asdict(state_before),
            state_after=state_after,
        )
        
    except Exception as e:
        logger.warning(f"Sample generation failed: {e}")
        return None


def generate_dataset(
    n_samples: int,
    output_path: Path,
    vllm_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    n_workers: int = 4,
):
    """Generate DSL training dataset"""
    
    client = create_llm_client(vllm_url)
    dsl = create_dsl()
    
    # Test connection
    try:
        models = client.models.list()
        available_model = models.data[0].id if models.data else model
        logger.info(f"Connected to vLLM, using model: {available_model}")
        model = available_model
    except Exception as e:
        logger.error(f"Cannot connect to vLLM: {e}")
        return
    
    samples = []
    parse_success = 0
    
    # Generate samples
    pbar = tqdm(total=n_samples, desc="Generating DSL samples")
    
    def worker(goal: str):
        sim = ParticleSimulator(width=64, height=64, max_particles=100)
        return generate_sample(client, model, dsl, sim, goal)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = []
        for i in range(n_samples):
            goal = random.choice(GOALS)
            futures.append(executor.submit(worker, goal))
        
        # Collect results
        for future in as_completed(futures):
            sample = future.result()
            if sample:
                samples.append(sample)
                if sample.parsed_successfully:
                    parse_success += 1
            pbar.update(1)
            pbar.set_postfix({
                'success': f"{parse_success}/{len(samples)}",
                'rate': f"{parse_success/max(len(samples),1):.1%}"
            })
    
    pbar.close()
    
    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample))+'\n')
    
    # Also save full JSON for inspection
    full_path = output_path.with_suffix('.full.json')
    with open(full_path, 'w') as f:
        json.dump([asdict(s) for s in samples], f, indent=2)
    
    logger.info(f"Generated {len(samples)} samples")
    logger.info(f"Parse success rate: {parse_success}/{len(samples)} ({parse_success/max(len(samples),1):.1%})")
    logger.info(f"Saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate DSL training data")
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/dsl_training.jsonl', help='Output path')
    parser.add_argument('--vllm-url', type=str, default='http://localhost:8000/v1', help='vLLM server URL')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct', help='Model name')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    generate_dataset(
        n_samples=args.n_samples,
        output_path=Path(args.output),
        vllm_url=args.vllm_url,
        model=args.model,
        n_workers=args.workers,
    )


if __name__ == '__main__':
    main()
