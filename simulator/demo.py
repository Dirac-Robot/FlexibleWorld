"""
Interactive demo for the particle simulator
Run with: python -m simulator.demo
"""
import numpy as np
import sys
from pathlib import Path

# try to import visualization libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from simulator.core import ParticleSimulator
from simulator.worlds.basic_physics import BasicPhysicsWorld, ZeroGravityWorld, InverseGravityWorld
from simulator.actions import ActionType


def run_headless_demo():
    """Run simulation without visualization (for testing)"""
    print('Running headless simulation demo...\n')

    worlds = [
        ('Basic Physics', BasicPhysicsWorld()),
        ('Zero Gravity', ZeroGravityWorld()),
        ('Inverse Gravity', InverseGravityWorld()),
    ]

    for name, world in worlds:
        print(f'=== {name} World ===')
        print(f'Config: {world.config.to_dict()}')

        sim = ParticleSimulator(world=world, width=64, height=64)

        # add particles
        sim.add_particle(x=16, y=16, vx=1, vy=0, color=(1, 0, 0))
        sim.add_particle(x=48, y=16, vx=-1, vy=0, color=(0, 0, 1))
        sim.add_agent(x=32, y=48)

        # simulate
        print('Running 50 steps...')
        for step in range(50):
            action = np.random.randint(0, 5)  # random movement
            sim.step(action=action)

        # final state
        state = sim.get_state()
        print(f'Final positions:')
        for i in range(state['n_particles']):
            ptype = 'agent' if state['types'][i] == 1 else 'particle'
            pos = state['positions'][i]
            print(f'  {ptype}: ({pos[0]:.1f}, {pos[1]:.1f})')
        print()

    print('✓ Headless demo completed')


def run_interactive_demo():
    """Run interactive demo with OpenCV visualization"""
    if not HAS_CV2:
        print('OpenCV not available. Install with: pip install opencv-python')
        print('Running headless demo instead...\n')
        run_headless_demo()
        return

    print('Interactive Particle Simulator Demo')
    print('===================================')
    print('Controls:')
    print('  W/A/S/D - Move agent')
    print('  SPACE   - Push particles')
    print('  E       - Pull particles')
    print('  R       - Spawn particle')
    print('  1/2/3   - Switch world (Basic/Zero-G/Inverse)')
    print('  ESC     - Quit')
    print()

    worlds = {
        ord('1'): ('Basic Physics', BasicPhysicsWorld()),
        ord('2'): ('Zero Gravity', ZeroGravityWorld()),
        ord('3'): ('Inverse Gravity', InverseGravityWorld()),
    }

    current_world = BasicPhysicsWorld()
    sim = None

    def reset_sim(world):
        nonlocal sim
        sim = ParticleSimulator(world=world, width=64, height=64)
        # add some initial particles
        for _ in range(10):
            x = np.random.uniform(5, 59)
            y = np.random.uniform(5, 59)
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.5, 0.5)
            color = tuple(np.random.uniform(0.5, 1.0, 3))
            sim.add_particle(x=x, y=y, vx=vx, vy=vy, color=color)
        sim.add_agent(x=32, y=32)

    reset_sim(current_world)

    key_to_action = {
        ord('w'): ActionType.MOVE_UP,
        ord('s'): ActionType.MOVE_DOWN,
        ord('a'): ActionType.MOVE_LEFT,
        ord('d'): ActionType.MOVE_RIGHT,
        ord(' '): ActionType.PUSH,
        ord('e'): ActionType.PULL,
        ord('r'): ActionType.SPAWN,
    }

    cv2.namedWindow('Particle Simulator', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Particle Simulator', 512, 512)

    while True:
        # render
        img = sim.render(scale=8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # add info text
        info = f'Step: {sim.step_count} | Particles: {sim.n_particles} | World: {sim.world.config.name}'
        cv2.putText(img_bgr, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Particle Simulator', img_bgr)

        # get key
        key = cv2.waitKey(50) & 0xFF

        if key == 27:  # ESC
            break
        elif key in worlds:
            name, world = worlds[key]
            print(f'Switching to {name}')
            current_world = world.__class__()
            reset_sim(current_world)
        elif key in key_to_action:
            action = key_to_action[key]
        else:
            action = ActionType.NOOP

        sim.step(action=action)

    cv2.destroyAllWindows()
    print('Demo ended')


if __name__ == '__main__':
    if '--headless' in sys.argv:
        run_headless_demo()
    else:
        run_interactive_demo()
