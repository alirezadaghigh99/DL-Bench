
from .techniques import BaseGenerator
class ZeroShotCoT(BaseGenerator):
    dl_bench_prompt = '''
    {prompt}
    Let's generate the code step by step.
    '''
    
    def __init__(self, dataset, model, tech_name):
        super().__init__(dataset, model, tech_name)
    
    def form_technique_prompt(self, prompt):
        return self.dl_bench_prompt.format(prompt=prompt)
    
    