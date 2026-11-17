class FitVisionError(Exception):
    pass


class RequestError(FitVisionError):
    pass


class CaptureError(FitVisionError):
    pass


class CalibrationError(FitVisionError):
    pass


class MeasurementError(FitVisionError):
    pass


class StudyError(FitVisionError):
    pass
