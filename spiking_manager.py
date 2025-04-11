from model_base import SNNModel

class SNNModelManager:
    def __init__(self):
        self.models = {}

    def add_model(self, model: SNNModel):
        if model.model_name in self.models:
            print(f"Model {model.model_name} already exists. Replacing it.")
        self.models[model.model_name] = model

    def run_model(self, model_name: str, input_data):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        model.load_model()
        processed_input = model.preprocess_input(input_data)
        output = model.run_inference(processed_input)
        return model.postprocess_output(output)
