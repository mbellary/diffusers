
__all__ = [
    "CancelFitException",
    "CancelEpochException",
    "CancelBatchException",
    "CancelPredictException",
    "CancelReportException",
    "CancelSaveException",
    "CancelResumeException",
    "CancelEvalException"
    
]

class CancelFitException(Exception)   : pass
class CancelEpochException(Exception) : pass
class CancelBatchException(Exception) : pass
class CancelPredictException(Exception) : pass
class CancelReportException(Exception)  : pass
class CancelSaveException(Exception)    : pass
class CancelResumeException(Exception)  : pass
class CancelEvalException(Exception)  : pass