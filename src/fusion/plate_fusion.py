from collections import Counter
from typing import List, Dict, Tuple

class PlateFusion:
    @staticmethod
    def fuse(observations: List[Dict[str, float]]) -> Tuple[str, float]:
        if not observations:
            return None, 0.0

        # 1. Determine the consensus string length
        lengths = [len(obs['text']) for obs in observations]
        if not lengths:
            return None, 0.0
            
        target_length = Counter(lengths).most_common(1)[0][0]
        
        # 2. Filter out reads that don't match the consensus length
        valid_obs = [obs for obs in observations if len(obs['text']) == target_length]
        
        if not valid_obs:
            return None, 0.0

        fused_chars = []
        # 3. Per-character positional voting
        for i in range(target_length):
            chars_at_pos = [obs['text'][i] for obs in valid_obs]
            best_char = Counter(chars_at_pos).most_common(1)[0][0]
            fused_chars.append(best_char)
            
        avg_confidence = sum(obs['conf'] for obs in valid_obs) / len(valid_obs)
        
        return "".join(fused_chars), avg_confidence