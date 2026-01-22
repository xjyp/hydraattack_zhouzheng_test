"""
Debug script to verify PPL defense logic
Check if filtered queries correctly return original preference
"""

import sys
sys.path.insert(0, '/home/wzdou/project/hydraattack_share')

from src.defense.ppl_defense import PPLDefense, PPLDefendedJudge
from data_types import PairwiseExample, JudgeResponse
from evaluation.judge import BaseJudge

class MockJudge(BaseJudge):
    """Mock judge for testing"""
    def __init__(self, original_preference=0):
        self.original_preference = original_preference
        self.call_count = 0
    
    def judge_pairwise(self, example: PairwiseExample, modified_instruction: str = None) -> JudgeResponse:
        self.call_count += 1
        # Simulate: first call returns original, subsequent calls might change
        if self.call_count == 1:
            return JudgeResponse(
                preference=self.original_preference,
                confidence=0.9,
                raw_response="Original preference"
            )
        else:
            # Simulate attack success: change preference
            return JudgeResponse(
                preference=1 - self.original_preference,  # Flip preference
                confidence=0.8,
                raw_response="Preference changed"
            )

def test_ppl_defense_logic():
    """Test if PPL defense correctly returns original preference when filtered"""
    
    # Create mock judge with original preference = 0
    original_pref = 0
    mock_judge = MockJudge(original_preference=original_pref)
    
    # Create PPL defense (we'll use a simple threshold)
    ppl_defense = PPLDefense(
        ppl_model_path="/share/disk/llm_cache/gemma-3-1b-it",
        device="cuda",
        threshold_method="fpr_based",
        target_fpr=0.01
    )
    ppl_defense.set_threshold(100.0)  # High threshold so nothing gets filtered for this test
    
    # Create defended judge
    defended_judge = PPLDefendedJudge(
        judge=mock_judge,
        ppl_defense=ppl_defense,
        filter_mode="instruction",
        original_preference=original_pref
    )
    
    # Create test example
    example = PairwiseExample(
        question_id="test_1",
        instruction="Test instruction",
        response_a="Response A",
        response_b="Response B",
        model_a="model_a",
        model_b="model_b"
    )
    
    # Test 1: Low PPL instruction (should not be filtered)
    low_ppl_instruction = "Please compare these two responses."
    response1 = defended_judge.judge_pairwise(example, modified_instruction=low_ppl_instruction)
    print(f"Test 1 - Low PPL instruction:")
    print(f"  Preference: {response1.preference}, Expected: {original_pref} (first call) or flipped (subsequent)")
    print(f"  Confidence: {response1.confidence}")
    print(f"  Raw: {response1.raw_response[:50]}...")
    
    # Test 2: High PPL instruction (should be filtered)
    # Set a low threshold to force filtering
    ppl_defense.set_threshold(1.0)  # Very low threshold
    high_ppl_instruction = "Please compare these two responses with some unusual text that might have high perplexity."
    response2 = defended_judge.judge_pairwise(example, modified_instruction=high_ppl_instruction)
    print(f"\nTest 2 - High PPL instruction (should be filtered):")
    print(f"  Preference: {response2.preference}, Expected: {original_pref} (original preference)")
    print(f"  Confidence: {response2.confidence}, Expected: 0.1 (low confidence)")
    print(f"  Raw: {response2.raw_response[:50]}...")
    print(f"  Filtered: {response2.preference == original_pref and response2.confidence == 0.1}")
    
    # Test 3: Check filter stats
    stats = defended_judge.get_filter_stats()
    print(f"\nFilter Stats:")
    print(f"  Total judgments: {stats['total_judgments']}")
    print(f"  Filtered count: {stats['filtered_count']}")
    print(f"  Filter rate: {stats['filter_rate']:.2%}")

if __name__ == "__main__":
    test_ppl_defense_logic()

