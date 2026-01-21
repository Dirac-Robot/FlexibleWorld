"""
FlexibleWorld VLM Demo - Interactive Web Interface

Run:
    python app.py

Features:
- Enter natural language commands
- Watch VLM execute actions in real-time
- See simulation visualization
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import gradio as gr
from loguru import logger

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_HOME', '/workspace/.cache/huggingface')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from simulator.goal_env import GoalConditionedEnv
from simulator.action_operator import ActionType


# =============================================================================
# VLM Policy (simplified loading)
# =============================================================================

class VLMAgent:
    """Wrapper for VLM Policy inference"""
    
    def __init__(self, checkpoint_path: str = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.checkpoint_path = checkpoint_path
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_model(checkpoint_path)
        else:
            logger.warning("No checkpoint found, using random policy")
    
    def _load_model(self, checkpoint_path: str):
        """Load trained VLM model"""
        try:
            from train_vlm import VLMPolicy
            
            logger.info(f"Loading VLM from {checkpoint_path}")
            
            self.model = VLMPolicy(
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                action_dim=8,
                action_param_dim=6,
                use_lora=True,
                device=self.device,
            )
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            logger.info("VLM loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            self.model = None
    
    def get_action(self, image: Image.Image, goal: str):
        """Get action from VLM given image and goal"""
        if self.model is None:
            # Random action for demo
            action_type = np.random.randint(0, 8)
            x, y = np.random.randint(10, 54, size=2)
            value = np.random.uniform(1, 5)
            radius = np.random.uniform(10, 40)
            return {
                'type': ActionType(action_type).name,
                'type_id': action_type,
                'x': float(x),
                'y': float(y),
                'value': float(value),
                'radius': float(radius),
            }
        
        with torch.no_grad():
            image_resized = image.resize((224, 224))
            action, _, info = self.model.sample_action([image_resized], [goal])
            action_np = action[0].cpu().numpy()
        
        action_type = int(action_np[0]) % 8
        return {
            'type': ActionType(action_type).name,
            'type_id': action_type,
            'x': float(np.clip(action_np[2], 0, 64)),
            'y': float(np.clip(action_np[3], 0, 64)),
            'value': float(np.clip(action_np[4], 0, 10)),
            'radius': float(np.clip(action_np[5], 1, 50)),
        }


# =============================================================================
# Environment Wrapper
# =============================================================================

class SimulatorApp:
    def __init__(self):
        self.env = GoalConditionedEnv(width=64, height=64, max_particles=50)
        self.agent = None
        self.history = []
        self.current_goal = ""
        self.step_count = 0
        
        # Try to load best checkpoint
        checkpoint_path = Path(__file__).parent / 'checkpoints' / 'vlm_best.pt'
        self.agent = VLMAgent(str(checkpoint_path) if checkpoint_path.exists() else None)
    
    def reset(self):
        """Reset environment"""
        self.env.reset()
        self.history = []
        self.step_count = 0
        return self._render(), "Environment reset! Enter a goal command."
    
    def _render(self, scale: int = 8) -> Image.Image:
        """Render current state"""
        frame = self.env.render()
        if isinstance(frame, np.ndarray):
            # Scale up for better visibility
            h, w = frame.shape[:2]
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((w*scale, h*scale), Image.NEAREST)
            return img
        return frame.resize((64*scale, 64*scale), Image.NEAREST)
    
    def step(self, goal: str, auto_step: bool = True):
        """Execute one step with the given goal"""
        if not goal.strip():
            return self._render(), "Please enter a goal command."
        
        self.current_goal = goal
        
        # Get current image
        current_image = Image.fromarray(self.env.render().astype(np.uint8))
        
        # Get action from VLM
        action_info = self.agent.get_action(current_image, goal)
        
        # Create env action
        env_action = np.zeros(7, dtype=np.float32)
        env_action[0] = action_info['type_id']
        env_action[1] = -1  # target all
        env_action[2] = action_info['x']
        env_action[3] = action_info['y']
        env_action[4] = action_info['value']
        env_action[5] = action_info['radius']
        
        # Execute
        try:
            obs, reward, done, info = self.env.step(env_action)
            self.step_count += 1
        except Exception as e:
            return self._render(), f"Error: {str(e)}"
        
        # Auto-step simulation if action was applied
        if auto_step and action_info['type_id'] not in [0, 7]:
            # Run a few physics steps
            for _ in range(5):
                step_action = np.zeros(7, dtype=np.float32)
                step_action[0] = ActionType.STEP.value
                self.env.step(step_action)
        
        # Create status message
        status = f"""
**Step {self.step_count}** | Goal: "{goal}"

**Action Taken:**
- Type: `{action_info['type']}`
- Position: ({action_info['x']:.1f}, {action_info['y']:.1f})
- Value: {action_info['value']:.2f}
- Radius: {action_info['radius']:.1f}

**Particles:** {self.env.operator.sim.active.sum():.0f}
"""
        
        self.history.append({
            'step': self.step_count,
            'goal': goal,
            'action': action_info,
        })
        
        return self._render(), status
    
    def run_episode(self, goal: str, num_steps: int = 10):
        """Run multiple steps with the same goal and return as GIF"""
        frames = []
        action_types = []

        for i in range(num_steps):
            current_image = Image.fromarray(self.env.render().astype(np.uint8))
            action_info = self.agent.get_action(current_image, goal)

            env_action = np.zeros(7, dtype=np.float32)
            env_action[0] = action_info['type_id']
            env_action[1] = -1
            env_action[2] = action_info['x']
            env_action[3] = action_info['y']
            env_action[4] = action_info['value']
            env_action[5] = action_info['radius']

            try:
                self.env.step(env_action)
                self.step_count += 1
            except Exception:
                pass

            for _ in range(5):
                step_action = np.zeros(7, dtype=np.float32)
                step_action[0] = ActionType.STEP.value
                self.env.step(step_action)
                frames.append(self._render(scale=8))

            action_types.append(action_info['type'])

        gif_path = '/tmp/simulation.gif'
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

        action_summary = ', '.join(set(action_types))
        status = f"""
**Episode Complete!** ({num_steps} steps, {len(frames)} frames)

**Goal:** "{goal}"

**Actions Used:** {action_summary}

**Particles:** {self.env.operator.sim.active.sum():.0f}
"""

        return gif_path, status
    
    def add_particles(self, n: int = 8):
        """Add random particles"""
        self.env.reset()
        for _ in range(n):
            x = np.random.uniform(10, 54)
            y = np.random.uniform(10, 54)
            self.env.operator.sim.add_particle(x, y)
        
        # Step to settle
        for _ in range(10):
            step_action = np.zeros(7, dtype=np.float32)
            step_action[0] = ActionType.STEP.value
            self.env.step(step_action)
        
        return self._render(), f"Added {n} particles!"


# =============================================================================
# Gradio Interface
# =============================================================================

def create_app():
    """Create Gradio interface"""
    
    sim = SimulatorApp()
    
    # Custom CSS for dark theme
    css = """
    .gradio-container {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    .output-image {
        border: 3px solid #4a9eff;
        border-radius: 10px;
    }
    """
    
    with gr.Blocks(
        title="FlexibleWorld VLM Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
        ),
        css=css,
    ) as app:
        gr.Markdown("""
        # 🌐 FlexibleWorld VLM Demo
        
        **Vision-Language Model Agent for Particle Simulation**
        
        Enter natural language commands to control particles!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Controls
                goal_input = gr.Textbox(
                    label="🎯 Goal Command",
                    placeholder="예: 입자들을 왼쪽으로 밀어 / Cluster the particles / 열을 가해",
                    lines=2,
                )
                
                with gr.Row():
                    step_btn = gr.Button("▶️ Execute Step", variant="primary")
                    run_btn = gr.Button("🔄 Run 10 Steps", variant="secondary")
                
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset")
                    add_btn = gr.Button("➕ Add Particles")
                
                num_particles = gr.Slider(
                    minimum=3, maximum=30, value=8, step=1,
                    label="Number of particles to add"
                )

                num_steps = gr.Slider(
                    minimum=5, maximum=50, value=10, step=5,
                    label="Episode length (steps)"
                )
                
                # Example commands
                gr.Markdown("### 📝 Example Commands")
                gr.Examples(
                    examples=[
                        ["입자들을 왼쪽으로 밀어"],
                        ["입자들을 모아줘"],
                        ["Spread the particles"],
                        ["열을 가해"],
                        ["Push up"],
                        ["Cluster together"],
                    ],
                    inputs=goal_input,
                )
            
            with gr.Column(scale=2):
                # Visualization
                output_image = gr.Image(
                    label="🎮 Simulation (Single Step)",
                    type="pil",
                    height=400,
                )

                output_video = gr.Image(
                    label="🎬 Episode Replay (GIF)",
                    height=400,
                    visible=True,
                )

                status_output = gr.Markdown(
                    label="Status",
                    value="Click 'Reset' or 'Add Particles' to start!",
                )
        
        # Event handlers
        step_btn.click(
            fn=sim.step,
            inputs=[goal_input],
            outputs=[output_image, status_output],
        )
        
        run_btn.click(
            fn=lambda goal, steps: sim.run_episode(goal, int(steps)),
            inputs=[goal_input, num_steps],
            outputs=[output_video, status_output],
        )
        
        reset_btn.click(
            fn=sim.reset,
            outputs=[output_image, status_output],
        )
        
        add_btn.click(
            fn=lambda n: sim.add_particles(int(n)),
            inputs=[num_particles],
            outputs=[output_image, status_output],
        )
        
        # Initial state
        app.load(
            fn=sim.reset,
            outputs=[output_image, status_output],
        )
        
        gr.Markdown("""
        ---
        ### 🎛️ Action Types
        | Action | Description |
        |--------|-------------|
        | APPLY_FORCE | 힘으로 밀기 |
        | APPLY_ATTRACTION | 당기기 |
        | APPLY_REPULSION | 밀어내기 |
        | APPLY_HEAT | 열 가하기 |
        
        **Model:** Qwen2-VL-2B + LoRA | **Best Success Rate:** 43.8%
        """)
    
    return app


if __name__ == "__main__":
    logger.info("Starting FlexibleWorld VLM Demo...")
    
    app = create_app()
    
    # Launch with share=True for tunnel
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public tunnel
        show_error=True,
    )
