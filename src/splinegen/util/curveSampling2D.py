
from opsIntersectDetect3D import sampleEqualChordLengthParam2D,sampleEqualChordLengthParam2DNoised
from opsIntersectDetect3D import sampleEqualChordLengthParam3D,sampleEqualChordLengthParam3DNoised
from util import curve
from util.curve import BSplineCurve
import numpy as np
import torch
from util.nurbs_eval import CurveEval

def EqualChordLenSamplingParams(c:curve.BSplineCurve,revolution:int=1000,size:int=1000,number=1)->tuple:
    return [sampleEqualChordLengthParam2D(c.degree,revolution,size,c.control_pts,c.knot_vector) for _ in range(number)]

def EqualChordLenSamplingParams3D(c:curve.BSplineCurve,revolution:int=1000,size:int=1000,number=1)->tuple:
    return [sampleEqualChordLengthParam3D(c.degree,revolution,size,c.control_pts,c.knot_vector) for _ in range(number)]

def NoisedEqualChordLenSamplingParams(c:curve.BSplineCurve,revolution:int=1000,size:int=1000,noise=0.02,eps=1e-7)->tuple:
    params=sampleEqualChordLengthParam2D(c.degree,revolution,size,c.control_pts,c.knot_vector)

    noise=np.random.normal(loc=0,scale=noise,size=len(params)-2)
    new_param=params[1:-1]+noise
    new_param=np.clip(new_param,eps,1-eps)
    params[1:-1]=new_param
    
    return params

def NoisedEqualChordLenSamplingParamsMulti(c:curve.BSplineCurve,revolution:int=1000,size:int=1000,noise=0.02,eps=1e-7,number=1)->tuple:
    return [NoisedEqualChordLenSamplingParams(c,revolution,size,noise,eps) for _ in range(number)]

def NoisedEqualChordLenSamplingParamsCpp(c:curve.BSplineCurve,revolution:int=1000,size:int=1000,noise=0.01,number=1)->tuple:
    return [sampleEqualChordLengthParam2DNoised(c.degree,revolution,size,noise,c.control_pts,c.knot_vector) for _ in range(number)]

def NoisedEqualChordLenSamplingParamsCpp3D(c:curve.BSplineCurve,revolution:int=1000,size:int=1000,noise=0.01,number=1)->tuple:
    return [sampleEqualChordLengthParam3DNoised(c.degree,revolution,size,noise,c.control_pts,c.knot_vector) for _ in range(number)]

from geomdl import BSpline

# def curveGradientVariation(curve:BSplineCurve,samples=1000,given_list=None,
#                           parameter_generator=EqualChordLenSamplingParams):
#     crv=BSpline.Curve()
#     crv.degree=curve.degree
#     crv.ctrlpts=curve.control_pts.tolist()
#     crv.knotvector=curve.knot_vector
    
#     if given_list is not None:
#         param_list_=given_list
#     else:
#         param_list_=parameter_generator(curve,size=samples)[0]
            
#     param_list=np.sort(param_list_)
       
#     # param_list=np.sorted(param_list_)

#     ders=[crv.derivatives(u,order=1) for u in param_list]
#     ders=np.array(ders)[:,1,:]
#     der_variation=np.var(ders,axis=0)
#     der_variation=np.sum(der_variation)

#     return der_variation
    
def curveGradientVariation(curve:BSplineCurve,samples=1000,given_list=None,
                          parameter_generator=EqualChordLenSamplingParams):
    crv=BSpline.Curve()
    crv.degree=curve.degree
    crv.ctrlpts=curve.control_pts.tolist()
    crv.knotvector=curve.knot_vector
    
    if given_list is not None:
        param_list_=given_list
    else:
        param_list_=parameter_generator(curve,size=samples)[0]
            
    param_list=np.sort(param_list_)
       
    # param_list=np.sorted(param_list_)
    ders=[crv.derivatives(u,order=2) for u in param_list]
    ders=np.array(ders)[:,2,:]
    ders=ders*ders
    ders=np.sum(ders,axis=-1)
    # der_variation=np.var(ders,axis=0)
    der_variation=np.mean(ders)

    return der_variation


def get_u_from_arc_length_torch(u_array,arc_array,s_array):
    # arc_table: [[t0,s0],[t1,s1],...,[tn,sn]]
    # s_array: [s0,s1,...,sn]
    # s in s_array \belongs [0,1]
    # return u in [0,1]
    total_length=arc_array[-1]
    # i=0

    u:torch.Tensor = s_array*total_length
    u=u.unsqueeze(0)
    uspan_uv = torch.min(torch.where((u - arc_array.unsqueeze(1))>1e-8, u - arc_array.unsqueeze(1), (u - arc_array.unsqueeze(1))*0.0 +total_length),0,keepdim=False)[1]

    return u_array[uspan_uv]

def get_arc_table_torch(ctrl_pts,knots,num_samples,degree=3):
    curve_evaluator=CurveEval(num_samples,p=degree)
    # u_array=np.linspace(0,1,num_samples)
    # points=deCasteljau_multple(ctrl_pts,u_array)
    ctrl_pts=ctrl_pts.squeeze().unsqueeze(0)
    knots=knots.squeeze().unsqueeze(0)

    points=curve_evaluator([ctrl_pts,knots])
    points=points.squeeze()
    # points=np.array(points)
    diff=points[1:]-points[:-1]
    diff=torch.linalg.norm(diff,dim=-1)
    diff=torch.cat([torch.zeros(1,device=diff.device),diff],dim=-1)
    arcs=torch.cumsum(diff,0)

    return curve_evaluator.u,arcs

def EqualChordLenSamplingParamsTorch(ctrl,knots,degree=3,revolution:int=1000,size:int=1000)->tuple:
    u_table,arc_table=get_arc_table_torch(ctrl_pts=ctrl,knots=knots,degree=degree,num_samples=revolution)

    return get_u_from_arc_length_torch(u_table,arc_table,torch.linspace(0,1,size,device=ctrl.device))