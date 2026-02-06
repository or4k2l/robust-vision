class ConfidenceMonitor:
    """Real-time confidence monitoring during inference."""
    
    def __init__(self, min_snr=10.0):
        self.min_snr = min_snr
        self.low_confidence_count = 0
    
    def check_prediction(self, logits):
        pred = logits.argmax()
        snr = logits[pred] / logits.max(where=logits!=logits[pred])
        
        if snr < self.min_snr:
            self.low_confidence_count += 1
            return {
                'prediction': pred,
                'confidence': 'LOW',
                'snr': float(snr),
                'action': 'REVIEW_REQUIRED'
            }
        
        return {
            'prediction': pred,
            'confidence': 'HIGH',
            'snr': float(snr),
            'action': 'PROCEED'
        }
