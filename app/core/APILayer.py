from typing import List
import numpy as np

from interview_seedtag import (
    PytorchClassifier, 
    SklearnClassifier
)

from schema import (
    GonkInput, 
    AstromechsInput, 
    Model,
    Output,
    Label
)

MAP = {
    0: "blue", 
    1: "green", 
    2: "yellow"
}
def probability_to_label(x: int) -> str:
        return MAP[x]

probability_to_label_vectorize = np.vectorize(probability_to_label)

class APILayer:
    def __init__(self, pytorch_classifier: PytorchClassifier, sklearn_classifier: SklearnClassifier):
        self.pytorch_classifier = pytorch_classifier
        self.sklearn_classifier = sklearn_classifier

    def __probabilities_to_labels__(self, scores: List[List[float]]) -> List[Label]:
        return probability_to_label_vectorize(
            np.argmax(scores, axis = 1)
        ).tolist()
    
    def sklearn(self, 
        input: GonkInput,
    ) -> Output:
        scores: List[List[float]] = self.sklearn_classifier.predict(input.crystalData)
        prediction: List[Label] = self.__probabilities_to_labels__(scores)
        return Output(
            prediction = prediction,
            scores = [{
                "blue": score[0],
                "green": score[1],
                "yellow": score[2]
            } for score in scores]
        )
    
    def pytorch(self, 
        input: GonkInput
    ) -> Output:
        scores: List[List[float]] = self.pytorch_classifier.predict(input.crystalData)
        prediction: List[Label] = self.__probabilities_to_labels__(scores)
        return Output(
            prediction = prediction,
            scores = [{
                "blue": score[0],
                "green": score[1],
                "yellow": score[2]
            } for score in scores]
        )

    def astromech(self, 
        input: AstromechsInput
    ) -> Output:
        if input.model == Model.sklearn:
            return self.sklearn(GonkInput(
                crystalData = input.crystalData
            ))
        elif input.model == Model.pytorch:
            return self.pytorch(GonkInput(
                crystalData = input.crystalData
            ))