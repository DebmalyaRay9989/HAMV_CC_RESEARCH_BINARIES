import numpy as np
from src.hamvcc.data import MultiViewData, apply_condition

def test_missing_condition():
    n=4; d=MultiViewData({'structural':np.ones((n,2)),'visual':np.ones((n,3)),'text':np.ones((n,4))},np.ones((n,3)))
    x=apply_condition(d,'no_text'); assert np.all(x.masks[:,2]==0)

def test_reject_zero_view():
    d=MultiViewData({'structural':np.ones((1,2)),'visual':np.ones((1,2)),'text':np.ones((1,2))},np.zeros((1,3)))
    try: d.validate(); assert False
    except ValueError: assert True
