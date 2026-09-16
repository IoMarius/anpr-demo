import cv2
import easyocr

from config import settings
from utils.device import log_gpu_status, is_cuda_oom, touch_cuda

class PlateRecognizer:
    def __init__(self, languages: list[str] = settings.recognition.languages,
                  gpu: bool = settings.recognition.gpu,
                  allowlist: str = settings.recognition.allowlist,
                  no_read_confidence: float = settings.recognition.no_read_confidence,
                   use_detector: bool = settings.recognition.easyocr_detector,
                   quantize: bool = settings.recognition.quantize,
                   min_ocr_conf: float = settings.recognition.min_ocr_conf):
        # Initialize EasyOCR, restricting to alphanumeric characters
        use_gpu = gpu and settings.gpu.enabled
        if use_gpu and not touch_cuda("cuda:0"):
            use_gpu = False
        self.reader = self._init_reader(languages, use_gpu,
                                        use_detector, quantize)
        self.use_gpu = self.reader.device == "cuda"
        self.allowlist = allowlist
        self.no_read_confidence = no_read_confidence
        self.min_ocr_conf = min_ocr_conf
        log_gpu_status(f"recognizer gpu={self.use_gpu} "
                       f"detector={use_detector} quantize={quantize}")

    @staticmethod
    def _init_reader(languages, use_gpu, use_detector, quantize):
        try:
            return easyocr.Reader(languages, gpu=use_gpu,
                                  detector=use_detector, quantize=quantize)
        except Exception as e:
            if use_gpu and is_cuda_oom(e):
                print(f"[OCR] GPU init failed ({str(e)[:150]}), "
                      f"falling back to CPU reader")
                return easyocr.Reader(languages, gpu=False,
                                      detector=use_detector,
                                      quantize=quantize)
            raise

    def recognize(self, plate_crop):
        # Upscale small crops: EasyOCR was trained on larger text;
        # 2x nearest-neighbor is ~0.1ms and lifts tiny-plate recall.
        h, w = plate_crop.shape[0], plate_crop.shape[1]
        if max(h, w) < 200:
            plate_crop = cv2.resize(plate_crop, (w * 2, h * 2),
                                    interpolation=cv2.INTER_CUBIC)
        results = self.reader.readtext(plate_crop, allowlist=self.allowlist)
        if not results:
            return None, self.no_read_confidence

        # In a tight plate crop, taking the highest confidence read is sufficient
        best_result = max(results, key=lambda x: x[2])
        text, confidence = best_result[1], best_result[2]
        if confidence < self.min_ocr_conf:
            return None, float(confidence)

        return text, float(confidence)

    def recognize_batch(self, plate_crops):
        """One call-site per frame for all OCR jobs.

        NOTE: EasyOCR exposes no true batched inference API, so crops are
        read sequentially. The scheduling (interval, top-K cap, confident
        stop) is what reduces OCR work; if a batched OCR engine replaces
        EasyOCR later, only this method needs to change. Benchmark first.
        """
        return [self.recognize(c) for c in plate_crops]