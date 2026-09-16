import re
from collections import Counter
from typing import List, Dict, Tuple

from config import settings

MIN_OBS_CONF = settings.recognition.min_ocr_conf
SINGLE_OBS_CONF = settings.recognition.stop_confidence
MIN_PLATE_LENGTH = settings.recognition.min_plate_length
PLATE_FORMAT = re.compile(settings.recognition.plate_format)


def normalize(text: str) -> str:
    """Strip separators/spaces, uppercase: 'ab 123' -> 'AB123'."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class PlateFusion:
    @staticmethod
    def fuse(observations: List[Dict[str, float]]) -> Tuple[str, float]:
        if not observations:
            return None, 0.0

        # 0. Drop low-confidence, too-short, and wrong-format reads so junk
        # OCR (e.g. STECBUD on a grille) cannot seed an event. Format is the
        # hard gate: MD plates are 3 letters + 3-4 digits.
        confident = []
        for obs in observations:
            text = normalize(obs.get("text", ""))
            if (float(obs.get("conf", 0.0)) >= MIN_OBS_CONF
                    and len(text) >= MIN_PLATE_LENGTH
                    and PLATE_FORMAT.fullmatch(text)):
                confident.append({"text": text, "conf": float(obs["conf"])})
        if not confident:
            return None, 0.0

        # 1. Determine the consensus string length
        lengths = [len(obs['text']) for obs in confident]
        if not lengths:
            return None, 0.0

        target_length = Counter(lengths).most_common(1)[0][0]

        # 2. Filter out reads that don't match the consensus length
        valid_obs = [obs for obs in confident if len(obs['text']) == target_length]

        if not valid_obs:
            return None, 0.0

        # 3. A lone observation needs high confidence to emit an event
        if len(valid_obs) == 1 and float(valid_obs[0]["conf"]) < SINGLE_OBS_CONF:
            return None, 0.0

        fused_chars = []
        # 3. Per-character positional voting
        for i in range(target_length):
            chars_at_pos = [obs['text'][i] for obs in valid_obs]
            best_char = Counter(chars_at_pos).most_common(1)[0][0]
            fused_chars.append(best_char)
            
        avg_confidence = sum(obs['conf'] for obs in valid_obs) / len(valid_obs)
        
        return "".join(fused_chars), avg_confidence