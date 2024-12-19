""""
test_uniti_con.py
this module is the unit test for uniti_con.py
expec_tim is the expected time given by the meda file its the system time start
"""
from uniti_con import gt_unix

EXPEC_TIM = 1729609903.749

def test_gt_unix():
    """This tests the imptomec function"""
    result = gt_unix('LL08_circle.md')
    assert result == EXPEC_TIM
