# =============================================================================
# Evaluation metrics
# =============================================================================
from sklearn.metrics import f1_score,classification_report
def acc_cal(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct
