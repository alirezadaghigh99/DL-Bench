

class BaseGenerator:
    def __init__(self, dataset, model, tech_name):
        self.ds = dataset
        self.model = model
        self.technique = tech_name
        self.system_message = "Only Generate python code"
        
    def form_technique_prompt(self, prompt):
        """
        This method should be overridden to generate the prompt for the specific technique.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_prompt(self, message):
        """
        This method should be overridden to generate the prompt for the specific technique.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_result(self, messages, data):
        """
        This method should be overridden to implement dataset-specific result generation.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def run_model(model_name):
        """
        This method should be overridden to run the model with the generated prompt.
        """
        raise NotImplementedError("Subclasses must implement this method.")