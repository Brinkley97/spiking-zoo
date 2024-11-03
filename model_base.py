from abc import ABC, abstractmethod

class SNNModel(ABC):
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None  # Placeholder for the actual loaded model

    @abstractmethod
    def load_model(self):
        """Method to load the SNN model."""
        pass

    @abstractmethod
    def preprocess_input(self, input_data):
        """Preprocess the input data for the model."""
        pass

    @abstractmethod
    def run_inference(self, input_data):
        """Run inference on the input data and return the result."""
        pass

    @abstractmethod
    def postprocess_output(self, output_data):
        """Postprocess the output data from the model."""
        pass
