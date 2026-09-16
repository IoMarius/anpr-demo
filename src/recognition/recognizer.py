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
                   min_ocr_conf: float = settings.recognition.min_ocr_conf,
                   upscale_target: int = settings.recognition.upscale_target,
                   upscale_max_factor: float = settings.recognition.upscale_max_factor,
                   sharpen: bool = settings.recognition.sharpen):
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
        self.upscale_target = upscale_target
        self.upscale_max_factor = upscale_max_factor
        self.sharpen = sharpen
        self.use_detector = use_detector and hasattr(self.reader,
                                                     "get_textbox")
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

    def _preprocess(self, plate_crop):
        # Adaptive upscale: far-camera plates can be ~40px wide while
        # EasyOCR resizes text lines to 64px height internally. Scale the
        # longest side toward the target (capped) so cubic interpolation
        # reconstructs stroke edges before recognition.
        h, w = plate_crop.shape[0], plate_crop.shape[1]
        if max(h, w) < self.upscale_target:
            factor = min(self.upscale_target / max(h, w),
                         self.upscale_max_factor)
            if factor > 1.0:
                plate_crop = cv2.resize(
                    plate_crop, (int(w * factor), int(h * factor)),
                    interpolation=cv2.INTER_CUBIC)
        if self.sharpen:
            # Mild unsharp mask against motion blur; reverted via config
            # if it ever amplifies sensor noise on tiny crops.
            blurred = cv2.GaussianBlur(plate_crop, (0, 0), 3)
            plate_crop = cv2.addWeighted(plate_crop, 1.5, blurred, -0.5, 0)
        return plate_crop

    def recognize(self, plate_crop):
        plate_crop = self._preprocess(plate_crop)
        if self.use_detector:
            results = self.reader.readtext(plate_crop,
                                           allowlist=self.allowlist)
        else:
            # Detector-less reader has no get_textbox, so readtext() would
            # crash; recognize() with no boxes defaults to a full-image
            # box, which is exactly right for a tight plate crop.
            results = self.reader.recognize(plate_crop,
                                            allowlist=self.allowlist)
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