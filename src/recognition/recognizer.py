import easyocr

from config import settings

class PlateRecognizer:
    def __init__(self, languages: list[str] = settings.recognition.languages,
                 gpu: bool = settings.recognition.gpu,
                 allowlist: str = settings.recognition.allowlist,
                 no_read_confidence: float = settings.recognition.no_read_confidence):
        # Initialize EasyOCR, restricting to alphanumeric characters
        self.reader = easyocr.Reader(languages, gpu=gpu)
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