# Group Analysis

def load_pkl():
    """
    Read in .pkl of Dorsiflexion or Plantar Flexion
    Muscle responses are already normalized to the maximum across all block of a participant
    .pkl are in a format of a list of dataframes SID_rest, SID_DF5p, SID_DF15p, SID_DF30p, SID_DF45p, SID_PF5p, SID_PF15p, SID_PF30p, SID_PF45p] 
    """

def get_restThresholds():
    """
    For individual participants, sort and save the rest threshold, and corresponding response
    Rest Threshold: the amplitude during the rest trial where there is first an increase in slope of the recruitment curve
        *where the amplitude reponse is no longer 0
    """

def get_noResponse():
    """
    For individual participants, sort and save the No-Response Threshold, and corresponding response
    No-Response Threshold: the last amplitude where the muscle response is still 0 (or <50 uV)
        *is the amplitude step just before the Rest Threshold
    """

def get_trialAUC(muscle, block):
    """
    Expore AUC for each muscle in a block, and for all blocks of a participant
    """
    #x_amplitudes = #amplitude list from participant
    #y_responses = #muscle responses of a block
    AUC = trapezoidal_AUC(x_amplitudes, y_responses)

def trapezoidal_AUC(x_amplitudes, y_responses):    
    """
    Calculate the area under a curve using the trapezoidal method.
    Parameters:
    - x: List or array of x-coordinates of the points
    - y: List or array of y-coordinates of the points
    Returns:
    - Area under the curve
    """
    if len(x) != len(y):
        raise ValueError("Input lists must have the same length")
    n = len(x)
    area = 0.0
    for i in range(1, n):
        # Trapezoidal rule formula: area = (h/2) * (y[i-1] + y[i])
        h = x[i] - x[i-1]
        area += (h / 2) * (y[i-1] + y[i])
    return area


def get_slopeThreshold():
    """
    Return slope between Rest Threshold point and No-Reponse Threshold point
    """

#TODO: setup barplots for comparisons of the above measures 