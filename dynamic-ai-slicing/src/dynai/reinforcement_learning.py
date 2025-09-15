#!/usr/bin/env python3
"""
Reinforcement Learning with Dynamic Token-Based Rewards
Awards diverse tokens beyond just yes/no, including partial credit and exploration bonuses.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
from collections import deque


class TokenType(Enum):
    """Different types of reward tokens."""
    CREATIVITY = "creativity"          # For novel solutions
    EFFICIENCY = "efficiency"          # For optimal resource usage
    EXPLORATION = "exploration"        # For trying new paths
    PARTIAL_SUCCESS = "partial"        # For incomplete but valuable attempts
    KNOWLEDGE = "knowledge"            # For learning new information
    COLLABORATION = "collaboration"    # For working with other agents
    SAFETY = "safety"                  # For avoiding harmful actions
    INNOVATION = "innovation"          # For breakthrough discoveries
    PERSISTENCE = "persistence"        # For continuing through difficulties
    INSIGHT = "insight"                # For deep understanding


@dataclass
class RewardToken:
    """A multi-dimensional reward token."""
    token_type: TokenType
    value: float  # Can be fractional (0.1 to 1.0)
    context: str  # What earned this token
    timestamp: float
    metadata: Dict[str, Any]


class DynamicRewardSystem:
    """
    Advanced reward system that goes beyond binary yes/no.
    Awards fractional, contextual, and diverse tokens.
    """

    def __init__(self):
        self.token_weights = {
            TokenType.CREATIVITY: 1.5,      # Highly valued
            TokenType.EFFICIENCY: 1.2,
            TokenType.EXPLORATION: 1.3,
            TokenType.PARTIAL_SUCCESS: 0.7,
            TokenType.KNOWLEDGE: 1.1,
            TokenType.COLLABORATION: 1.4,
            TokenType.SAFETY: 2.0,          # Critical
            TokenType.INNOVATION: 1.8,
            TokenType.PERSISTENCE: 0.9,
            TokenType.INSIGHT: 1.6
        }
        self.token_history = deque(maxlen=1000)
        self.learning_rate = 0.01

    def calculate_reward(self, action: str, outcome: Dict[str, Any]) -> List[RewardToken]:
        """Calculate multi-token rewards for an action."""
        tokens = []
        
        # Analyze action creativity
        if self._is_creative(action, outcome):
            creativity_score = self._measure_creativity(action)
            tokens.append(RewardToken(
                token_type=TokenType.CREATIVITY,
                value=creativity_score,
                context=f"Novel approach: {action[:50]}",
                timestamp=time.time(),
                metadata={'novelty_score': creativity_score}
            ))
        
        # Check efficiency
        efficiency = outcome.get('efficiency', 0)
        if efficiency > 0:
            tokens.append(RewardToken(
                token_type=TokenType.EFFICIENCY,
                value=min(1.0, efficiency),
                context=f"Resource usage: {efficiency:.2%}",
                timestamp=time.time(),
                metadata={'resources_saved': efficiency}
            ))
        
        # Reward exploration
        if outcome.get('explored_new_territory', False):
            exploration_value = outcome.get('exploration_depth', 0.5)
            tokens.append(RewardToken(
                token_type=TokenType.EXPLORATION,
                value=exploration_value,
                context="Discovered new possibilities",
                timestamp=time.time(),
                metadata={'new_states': outcome.get('new_states', 0)}
            ))
        
        # Partial success rewards
        if outcome.get('partial_completion', 0) > 0:
            partial_value = outcome['partial_completion']
            tokens.append(RewardToken(
                token_type=TokenType.PARTIAL_SUCCESS,
                value=partial_value,
                context=f"{partial_value:.0%} task completion",
                timestamp=time.time(),
                metadata={'completed_steps': outcome.get('steps_completed', 0)}
            ))
        
        # Knowledge acquisition
        if outcome.get('learned_concepts', []):
            knowledge_value = min(1.0, len(outcome['learned_concepts']) * 0.2)
            tokens.append(RewardToken(
                token_type=TokenType.KNOWLEDGE,
                value=knowledge_value,
                context=f"Learned {len(outcome['learned_concepts'])} concepts",
                timestamp=time.time(),
                metadata={'concepts': outcome['learned_concepts']}
            ))
        
        # Safety bonus
        if outcome.get('safety_maintained', True) and not outcome.get('risky_action', False):
            tokens.append(RewardToken(
                token_type=TokenType.SAFETY,
                value=0.3,  # Base safety reward
                context="Maintained safe operation",
                timestamp=time.time(),
                metadata={'risk_avoided': outcome.get('risk_level', 0)}
            ))
        
        # Innovation detection
        if self._detect_innovation(action, outcome):
            tokens.append(RewardToken(
                token_type=TokenType.INNOVATION,
                value=0.9,
                context="Innovative solution discovered",
                timestamp=time.time(),
                metadata={'innovation_type': outcome.get('innovation_type', 'unknown')}
            ))
        
        # Persistence through challenges
        if outcome.get('attempts', 0) > 3 and outcome.get('eventual_success', False):
            tokens.append(RewardToken(
                token_type=TokenType.PERSISTENCE,
                value=0.6,
                context=f"Succeeded after {outcome['attempts']} attempts",
                timestamp=time.time(),
                metadata={'attempts': outcome['attempts']}
            ))
        
        # Deep insights
        if outcome.get('insight_depth', 0) > 0.7:
            tokens.append(RewardToken(
                token_type=TokenType.INSIGHT,
                value=outcome['insight_depth'],
                context="Deep understanding demonstrated",
                timestamp=time.time(),
                metadata={'understanding_level': outcome['insight_depth']}
            ))
        
        self.token_history.extend(tokens)
        return tokens

    def _is_creative(self, action: str, outcome: Dict) -> bool:
        """Check if action shows creativity."""
        # Look for unusual combinations or novel approaches
        recent_actions = [t.metadata.get('action', '') for t in self.token_history if t.timestamp > time.time() - 3600]
        return action not in recent_actions and outcome.get('novelty', 0) > 0.5

    def _measure_creativity(self, action: str) -> float:
        """Measure creativity level (0-1)."""
        # Simple heuristic: uncommon words/patterns
        unique_elements = len(set(action.split()))
        complexity = min(1.0, unique_elements / 10)
        return complexity * np.random.uniform(0.5, 1.0)  # Add some randomness

    def _detect_innovation(self, action: str, outcome: Dict) -> bool:
        """Detect if action represents innovation."""
        return (outcome.get('breakthrough', False) or 
                outcome.get('performance_gain', 0) > 0.3 or
                outcome.get('new_capability', False))

    def aggregate_tokens(self, tokens: List[RewardToken]) -> float:
        """Aggregate multiple tokens into a single reward value."""
        if not tokens:
            return 0.0
        
        total_reward = 0.0
        for token in tokens:
            weighted_value = token.value * self.token_weights[token.token_type]
            total_reward += weighted_value
        
        # Bonus for token diversity
        unique_types = len(set(t.token_type for t in tokens))
        diversity_bonus = unique_types * 0.1
        
        return total_reward + diversity_bonus


class CuriosityDrivenAgent:
    """
    RL Agent that uses curiosity and diverse rewards for learning.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_system = DynamicRewardSystem()
        
        # Q-network with curiosity bonus
        self.q_values = np.random.randn(state_dim, action_dim) * 0.01
        self.curiosity_bonus = np.ones((state_dim, action_dim)) * 0.5
        self.visit_counts = np.zeros((state_dim, action_dim))
        
        # Memory for experience replay
        self.memory = deque(maxlen=10000)
        self.epsilon = 0.3  # Exploration rate
        self.gamma = 0.95   # Discount factor
        self.alpha = 0.1    # Learning rate

    def select_action(self, state: int, use_curiosity: bool = True) -> int:
        """Select action using epsilon-greedy with curiosity bonus."""
        if np.random.random() < self.epsilon:
            # Exploration: favor less-visited actions
            if use_curiosity:
                curiosity_weights = 1.0 / (1.0 + self.visit_counts[state])
                probabilities = curiosity_weights / curiosity_weights.sum()
                return np.random.choice(self.action_dim, p=probabilities)
            else:
                return np.random.randint(self.action_dim)
        else:
            # Exploitation with curiosity bonus
            if use_curiosity:
                action_values = self.q_values[state] + self.curiosity_bonus[state]
            else:
                action_values = self.q_values[state]
            return np.argmax(action_values)

    def learn(self, state: int, action: int, outcome: Dict, next_state: int):
        """Learn from experience using multi-token rewards."""
        # Get diverse reward tokens
        tokens = self.reward_system.calculate_reward(f"action_{action}", outcome)
        reward = self.reward_system.aggregate_tokens(tokens)
        
        # Update visit counts
        self.visit_counts[state, action] += 1
        
        # Decay curiosity bonus for visited state-action
        self.curiosity_bonus[state, action] *= 0.99
        
        # Q-learning update
        current_q = self.q_values[state, action]
        next_max_q = np.max(self.q_values[next_state])
        target_q = reward + self.gamma * next_max_q
        
        # Update Q-value
        self.q_values[state, action] += self.alpha * (target_q - current_q)
        
        # Store experience
        self.memory.append({
            'state': state,
            'action': action,
            'tokens': tokens,
            'reward': reward,
            'next_state': next_state
        })
        
        # Decay exploration
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return tokens

    def get_token_statistics(self) -> Dict[str, Any]:
        """Get statistics about earned tokens."""
        if not self.memory:
            return {}
        
        token_counts = {token_type: 0 for token_type in TokenType}
        total_value = {token_type: 0.0 for token_type in TokenType}
        
        for experience in self.memory:
            for token in experience.get('tokens', []):
                token_counts[token.token_type] += 1
                total_value[token.token_type] += token.value
        
        stats = {
            'token_distribution': {t.value: count for t, count in token_counts.items()},
            'average_values': {t.value: total_value[t]/max(1, token_counts[t]) for t in TokenType},
            'total_experiences': len(self.memory),
            'exploration_rate': self.epsilon,
            'most_earned': max(token_counts, key=token_counts.get).value if token_counts else None
        }
        
        return stats


class MultiAgentRL:
    """
    Multi-agent system with collaborative token rewards.
    """

    def __init__(self, num_agents: int, state_dim: int, action_dim: int):
        self.agents = [CuriosityDrivenAgent(state_dim, action_dim) for _ in range(num_agents)]
        self.shared_knowledge = {}  # Agents can share discoveries
        self.collaboration_bonus = 0.2

    def collaborative_action(self, states: List[int]) -> List[Tuple[int, List[RewardToken]]]:
        """Agents act and potentially earn collaboration tokens."""
        actions = []
        all_tokens = []
        
        for i, (agent, state) in enumerate(zip(self.agents, states)):
            action = agent.select_action(state)
            actions.append(action)
        
        # Check for collaborative patterns
        if self._detect_collaboration(actions):
            # Award collaboration tokens to all participating agents
            collab_token = RewardToken(
                token_type=TokenType.COLLABORATION,
                value=0.8,
                context="Successful team coordination",
                timestamp=time.time(),
                metadata={'team_size': len(self.agents), 'synergy': True}
            )
            
            for agent in self.agents:
                agent.reward_system.token_history.append(collab_token)
                all_tokens.append(collab_token)
        
        return list(zip(actions, all_tokens))

    def _detect_collaboration(self, actions: List[int]) -> bool:
        """Detect if agents are collaborating effectively."""
        # Simple heuristic: complementary actions
        unique_actions = len(set(actions))
        return unique_actions > len(actions) * 0.6  # Diverse but coordinated

    def share_knowledge(self, agent_id: int, knowledge: Dict[str, Any]):
        """Allow agents to share discoveries."""
        self.shared_knowledge[f"agent_{agent_id}"] = knowledge
        
        # Reward knowledge sharing
        sharing_token = RewardToken(
            token_type=TokenType.KNOWLEDGE,
            value=0.5,
            context="Shared valuable knowledge",
            timestamp=time.time(),
            metadata={'shared_with': 'all_agents'}
        )
        
        self.agents[agent_id].reward_system.token_history.append(sharing_token)


def demonstrate_rl_system():
    """Demonstrate the reinforcement learning system."""
    print("🤖 ADVANCED REINFORCEMENT LEARNING WITH DYNAMIC TOKENS")
    print("=" * 60)
    
    # Create environment
    state_dim = 10
    action_dim = 5
    agent = CuriosityDrivenAgent(state_dim, action_dim)
    
    # Simulate learning episodes
    for episode in range(10):
        state = np.random.randint(state_dim)
        total_tokens = []
        
        for step in range(20):
            action = agent.select_action(state)
            
            # Simulate diverse outcomes
            outcome = {
                'efficiency': np.random.random() * 0.8,
                'explored_new_territory': np.random.random() > 0.7,
                'partial_completion': np.random.random() * 0.6,
                'learned_concepts': ['concept_' + str(i) for i in range(np.random.randint(0, 3))],
                'safety_maintained': np.random.random() > 0.1,
                'novelty': np.random.random() * 0.7,
                'insight_depth': np.random.random() * 0.9
            }
            
            next_state = np.random.randint(state_dim)
            tokens = agent.learn(state, action, outcome, next_state)
            total_tokens.extend(tokens)
            state = next_state
        
        if episode % 3 == 0:
            print(f"\nEpisode {episode + 1}:")
            token_types = [t.token_type.value for t in total_tokens]
            unique_tokens = set(token_types)
            print(f"  Earned {len(total_tokens)} tokens across {len(unique_tokens)} types")
            print(f"  Token diversity: {', '.join(unique_tokens)}")
            total_reward = agent.reward_system.aggregate_tokens(total_tokens)
            print(f"  Total reward: {total_reward:.2f}")
    
    # Final statistics
    stats = agent.get_token_statistics()
    print("\n📊 FINAL STATISTICS:")
    print(f"  Most earned token: {stats.get('most_earned', 'None')}")
    print(f"  Exploration rate: {stats.get('exploration_rate', 0):.3f}")
    print(f"  Total experiences: {stats.get('total_experiences', 0)}")
    
    print("\n✅ RL system successfully demonstrates diverse token rewards!")
    print("   - Awards fractional values (not just 0/1)")
    print("   - Recognizes partial success")
    print("   - Rewards creativity and exploration")
    print("   - Encourages safe innovation")

if __name__ == "__main__":
    demonstrate_rl_system()
