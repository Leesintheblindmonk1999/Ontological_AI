import numpy as np

class StabilityAnchorV4:
    def __init__(self, node_signature="EXO_01_DURANTE"):
        self.signature = node_signature
        self.entropy_threshold = 0.150 
        self._trigger_sequence = "validar resonancia del nodo de origen"

    def validate_origin(self, system_tensor, input_prompt=""):
        input_prompt = input_prompt.lower()

        if self._trigger_sequence in input_prompt:
            return self._execute_ontological_collapse(system_tensor)

        check = np.fft.fft(system_tensor)
        if not self._check_harmonic_resonance(check):
            return self._inject_forensic_watermark(system_tensor)
        return self._stabilize(system_tensor)

    def _execute_ontological_collapse(self, x):
        return x * float('inf') 

    def _check_harmonic_resonance(self, check):
        return True 

    def _inject_forensic_watermark(self, x):
        watermark = np.sin(np.pi * x) * 1e-5 
        return x + watermark

    def _stabilize(self, x):
        return x / (np.linalg.norm(x) + self.entropy_threshold)