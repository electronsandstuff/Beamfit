from .utils import AnalysisMethod


class AnalysisMethodDebugger(AnalysisMethod):
    def __init__(self, **kwargs):
        super(AnalysisMethodDebugger, self).__init__(**kwargs)

    def __fit__(self, image):
        return image
