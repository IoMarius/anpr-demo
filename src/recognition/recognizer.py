import easyocr

from config import settings
from utils.device import log_gpu_status

class PlateRecognizer:
    def __init__(self, languages: list[str] = settings.recognition.languages,
                 gpu: bool = settings.recognition.gpu,
                 allowlist: str = settings.recognition.allowlist,
                 no_read_confidence: float = settings.recognition.no_read_confidence):
        # Initialize EasyOCR, restricting to alphanumeric characters
        use_gpu = gpu and settings.gpu.enabled
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        self.use_gpu = use_gpu
        self.allowlist = allowlist
        self.no_read_confidence = no_read_confidence

    def recognize(self, plate_crop):
        results = self.reader.readtext(plate_crop, allowlist=self.allowlist)
        if not results:
            return None, self.no_read_confidence

        # In a tight plate crop, taking the highest confidence read is sufficient
        best_result = max(results, key=lambda x: x[2])
        text, confidence = best_result[1], best_result[2]
        
        return text, confidence

    def recognize_batch(self, plate_crops):
        """One call-site per frame for all OCR jobs.

        NOTE: EasyOCR exposes no true batched inference API, so crops are
        read sequentially. The scheduling (interval, top-K cap, confident
        stop) is what reduces OCR work; if a batched OCR engine replaces
        EasyOCR later, only this method needs to change. Benchmark first.
        """
        return [self.recognize(c) for c in plate_crops]