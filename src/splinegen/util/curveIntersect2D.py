from .curve import BSplineCurve
from opsIntersectDetect3D import isIntersect2D,isIntersect

def isSelfIntersect(c:BSplineCurve,restrict=True,eps=1e-3,scale=1000,numSamples=1000,use_equal_chord=True)->bool:
    if c.isSelfIntersecting(restrict=restrict,eps=eps):
        return True
    # before calling isIntersect2D, the curve must be normalized to [0,1]
    return isIntersect2D(c.degree,scale,numSamples,use_equal_chord,c.control_pts+.5,c.knot_vector)

def isSelfIntersect3D(c:BSplineCurve,restrict=True,eps=1e-3,scale=1000,numSamples=1000,use_equal_chord=True)->bool:
    c=c.getNormalized(new_center=0.5)
    if c.isSelfIntersecting(restrict=restrict,eps=eps):
        return True
    # before calling isIntersect2D, the curve must be normalized to [0,1]
    return isIntersect(c.degree,scale,numSamples,use_equal_chord,c.control_pts,c.knot_vector)