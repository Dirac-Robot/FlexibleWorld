"""
LLM-based Data Augmentation for Goal-Conditioned Simulation
Generates diverse natural language commands for predefined goals
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from enum import Enum
import json
from pathlib import Path

from simulator.goal_env import GoalType


@dataclass
class GoalDefinition:
    """Structured goal definition for augmentation"""
    goal_type: GoalType
    description: str  # human-readable description
    base_commands: List[str] = field(default_factory=list)  # seed commands


# Predefined goals with base commands
GOAL_DEFINITIONS = [
    GoalDefinition(
        GoalType.LEFT,
        "입자들을 왼쪽으로 이동시키기",
        ["왼쪽으로", "move left", "좌측 이동"]
    ),
    GoalDefinition(
        GoalType.RIGHT,
        "입자들을 오른쪽으로 이동시키기", 
        ["오른쪽으로", "move right", "우측 이동"]
    ),
    GoalDefinition(
        GoalType.UP,
        "입자들을 위로 이동시키기",
        ["위로", "move up", "상단으로"]
    ),
    GoalDefinition(
        GoalType.DOWN,
        "입자들을 아래로 이동시키기",
        ["아래로", "move down", "하단으로"]
    ),
    GoalDefinition(
        GoalType.CENTER,
        "입자들을 중앙으로 모으기",
        ["가운데로", "to center", "중앙 집결"]
    ),
    GoalDefinition(
        GoalType.CLUSTER,
        "입자들을 한 곳에 뭉치기",
        ["모아줘", "cluster", "뭉쳐"]
    ),
    GoalDefinition(
        GoalType.SCATTER,
        "입자들을 흩어지게 하기",
        ["흩어지게", "scatter", "분산"]
    ),
    GoalDefinition(
        GoalType.EXPLODE,
        "입자들을 폭발시키기",
        ["폭발시켜", "explode", "터뜨려"]
    ),
    GoalDefinition(
        GoalType.HEAT,
        "입자들을 가열하기",
        ["가열해", "heat up", "뜨겁게"]
    ),
    GoalDefinition(
        GoalType.COOL,
        "입자들을 냉각하기",
        ["식혀줘", "cool down", "차갑게"]
    ),
]


def build_augmentation_prompt(goal: GoalDefinition, n_samples: int = 10) -> str:
    """Build prompt for LLM augmentation"""
    base_examples = "\n".join(f"  - {cmd}" for cmd in goal.base_commands)
    
    prompt = f"""당신은 2D 입자 시뮬레이션을 제어하는 사용자입니다.

목표: {goal.description}
목표 타입: {goal.goal_type.value}

기존 명령 예시:
{base_examples}

이 목표를 달성하기 위해 시뮬레이터에 명령하는 다양한 표현을 {n_samples}개 생성해주세요.

규칙:
1. 한국어와 영어를 섞어서 생성
2. 캐주얼한 표현 (예: "왼쪽으로 ㄱㄱ")
3. 정중한 표현 (예: "왼쪽으로 이동시켜 주세요")
4. 짧은 표현 (예: "좌")
5. 긴 설명형 (예: "모든 입자를 화면 왼쪽 끝으로 이동시켜줘")
6. 기존 예시와 중복되지 않게

JSON 형식으로 출력:
{{"commands": ["표현1", "표현2", ...]}}"""
    
    return prompt


class DataAugmentor:
    """
    LLM-based data augmentation for goal commands.
    
    Usage:
        # With API function
        def call_llm(prompt: str) -> str:
            return openai.chat.completions.create(...).choices[0].message.content
        
        augmentor = DataAugmentor(llm_fn=call_llm)
        augmentor.augment_all()
        augmentor.save('augmented_data.json')
    """

    def __init__(self, 
                 llm_fn: Optional[Callable[[str], str]] = None,
                 goals: List[GoalDefinition] = None):
        self.llm_fn = llm_fn or self._mock_llm
        self.goals = goals or GOAL_DEFINITIONS
        self.augmented_data: Dict[str, List[str]] = {}

    def _mock_llm(self, prompt: str) -> str:
        """Mock LLM for testing without API"""
        # Extract goal type from prompt
        for goal in self.goals:
            if goal.description in prompt:
                # Generate simple variations
                variations = []
                for base in goal.base_commands:
                    variations.extend([
                        f"{base}!",
                        f"{base} 해줘",
                        f"please {base}",
                        f"{base} 좀",
                    ])
                return json.dumps({"commands": variations[:10]})
        
        return json.dumps({"commands": ["default command"]})

    def augment_goal(self, goal: GoalDefinition, n_samples: int = 10) -> List[str]:
        """Augment a single goal with LLM"""
        prompt = build_augmentation_prompt(goal, n_samples)
        
        response = self.llm_fn(prompt)
        
        try:
            # Parse JSON response
            data = json.loads(response)
            commands = data.get("commands", [])
        except json.JSONDecodeError:
            # Try to extract commands from text
            commands = []
        
        # Combine with base commands
        all_commands = list(set(goal.base_commands + commands))
        
        return all_commands

    def augment_all(self, n_samples: int = 10, progress: bool = True) -> Dict[str, List[str]]:
        """Augment all goals"""
        self.augmented_data = {}
        
        for i, goal in enumerate(self.goals):
            if progress:
                print(f"Augmenting {goal.goal_type.value} ({i+1}/{len(self.goals)})")
            
            commands = self.augment_goal(goal, n_samples)
            self.augmented_data[goal.goal_type.value] = commands
        
        return self.augmented_data

    def save(self, path: str = 'augmented_goals.json'):
        """Save augmented data"""
        output = {
            "goals": [
                {
                    "goal_type": goal.goal_type.value,
                    "description": goal.description,
                    "commands": self.augmented_data.get(goal.goal_type.value, goal.base_commands)
                }
                for goal in self.goals
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Saved to {path}")
        return path

    def get_training_pairs(self) -> List[Dict]:
        """Get (command, goal_type) pairs for training"""
        pairs = []
        for goal_type, commands in self.augmented_data.items():
            for cmd in commands:
                pairs.append({
                    "input": cmd,
                    "output": goal_type,
                })
        return pairs

    def save_training_data(self, path: str = 'goal_classification_train.jsonl'):
        """Save as training data for goal classification"""
        pairs = self.get_training_pairs()
        
        with open(path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(pairs)} training pairs to {path}")
        return path


# OpenAI API integration example
def create_openai_augmentor(api_key: str = None) -> DataAugmentor:
    """Create augmentor with OpenAI API"""
    try:
        import openai
        if api_key:
            openai.api_key = api_key
        
        def llm_fn(prompt: str) -> str:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            return response.choices[0].message.content
        
        return DataAugmentor(llm_fn=llm_fn)
    
    except ImportError:
        print("OpenAI not installed. Using mock LLM.")
        return DataAugmentor()


# Gemini API integration example  
def create_gemini_augmentor(api_key: str = None) -> DataAugmentor:
    """Create augmentor with Gemini API"""
    try:
        import google.generativeai as genai
        if api_key:
            genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        def llm_fn(prompt: str) -> str:
            response = model.generate_content(prompt)
            return response.text
        
        return DataAugmentor(llm_fn=llm_fn)
    
    except ImportError:
        print("Google AI not installed. Using mock LLM.")
        return DataAugmentor()


# Claude API integration
def create_claude_augmentor(api_key: str = None) -> DataAugmentor:
    """Create augmentor with Claude API (claude-sonnet-4-5-20250514)"""
    try:
        import anthropic
        import os
        
        key = api_key or os.getenv('ANTHROPIC_API_KEY')
        client = anthropic.Anthropic(api_key=key)
        
        def llm_fn(prompt: str) -> str:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        
        return DataAugmentor(llm_fn=llm_fn)
    
    except ImportError:
        print("Anthropic not installed. pip install anthropic")
        return DataAugmentor()
    except Exception as e:
        print(f"Claude error: {e}")
        return DataAugmentor()


# LangChain universal integration
def create_langchain_augmentor(model_name: str = "claude-sonnet-4-5-20250514") -> DataAugmentor:
    """
    Create augmentor with any LangChain-supported LLM.
    
    Usage:
        # Claude
        augmentor = create_langchain_augmentor("claude-sonnet-4-5-20250514")
        
        # OpenAI
        augmentor = create_langchain_augmentor("gpt-4o-mini")
        
        # Gemini
        augmentor = create_langchain_augmentor("gemini-1.5-flash")
    """
    try:
        from langchain_core.messages import HumanMessage
        import os
        
        # Detect provider from model name
        if "claude" in model_name.lower():
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model_name, temperature=0.8)
        elif "gpt" in model_name.lower():
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model_name, temperature=0.8)
        elif "gemini" in model_name.lower():
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.8)
        else:
            # Default to Claude
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model_name, temperature=0.8)
        
        def llm_fn(prompt: str) -> str:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        return DataAugmentor(llm_fn=llm_fn)
    
    except ImportError as e:
        print(f"LangChain not installed: {e}")
        print("pip install langchain-anthropic langchain-openai langchain-google-genai")
        return DataAugmentor()
    except Exception as e:
        print(f"LangChain error: {e}")
        return DataAugmentor()


