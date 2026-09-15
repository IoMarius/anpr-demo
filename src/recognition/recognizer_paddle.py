# from paddleocr import PaddleOCR
# from config import settings

# class PlateRecognizer:
#     def __init__(self, 
#                  lang: str = settings.recognition.lang,
#                  gpu: bool = settings.recognition.gpu,
#                  allowlist: str = settings.recognition.allowlist,
#                  no_read_confidence: float = settings.recognition.no_read_confidence,
#                  show_log: bool = settings.recognition.show_log):
        
#         # Initialize PaddleOCR (cls=False disables text angle classification to save time)
#         self.reader = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=gpu, show_log=show_log)
#         self.allowlist = set(allowlist)
#         self.no_read_confidence = no_read_confidence

#     def recognize(self, plate_crop):
#         results = self.reader.ocr(plate_crop, cls=False)
        
#         # PaddleOCR returns None or [None] if nothing is found
#         if not results or not results[0]:
#             return None, self.no_read_confidence

#         best_text = ""
#         max_conf = 0.0

#         # results[0] contains a list of [box, (text, confidence)]
#         for box, (text, confidence) in results[0]:
#             # Force uppercase and strip characters not in the allowlist
#             filtered_text = "".join([c for c in text.upper() if c in self.allowlist])
            
#             if filtered_text and confidence > max_conf:
#                 best_text = filtered_text
#                 max_conf = float(confidence)

#         if not best_text:
#             return None, self.no_read_confidence
            
#         return best_text, max_conf