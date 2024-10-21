import numpy as np
from geomdl import BSpline
from geomdl.exceptions import GeomdlException
from geomdl import helpers
from util.curve import BSplineCurve,BSplineCurveGenerator,samplePointsFromCurve
from scipy.optimize import brent
# from util.curve import Curve

def getBasisFuncs(degree,knot_vector,knots):
    # helpers.basis_functions(degree,knot_vector,)
    spans=helpers.find_spans(degree,knot_vector,len(knot_vector)-degree-1,knots)
    return helpers.basis_functions(degree,knot_vector,spans,knots)

def getBasisFuncsFromCurve(curve:BSplineCurve,knots):
    # helpers.basis_functions(degree,knot_vector,)
    degree=curve.degree
    knot_vector=curve.knot_vector
    spans=helpers.find_spans(degree,knot_vector,len(knot_vector)-degree-1,knots)
    return helpers.basis_functions(degree,knot_vector,spans,knots)


def getBasisFuncsAll(degree,knot_vector,knots):
    spans=helpers.find_spans(degree,knot_vector,len(knot_vector)-degree-1,knots)
    basis = helpers.basis_functions(degree,knot_vector,spans,knots)
    basis_=np.zeros((len(knots),len(knot_vector)-degree-1))
    for i in range(len(knots)):
        basis_[i,spans[i]-degree:spans[i]+1]=basis[i]
    return basis_

def getBasisFuncsAllFromCurve(curve:BSplineCurve,knots):
    # helpers.basis_functions(degree,knot_vector,)
    degree=curve.degree
    knot_vector=curve.knot_vector
    return getBasisFuncsAll(degree,knot_vector,knots)

def approximateCtrlPts(degree,knot_vector,points,params):
    '''
        return ctrl_pts,loss
    '''
    basis=getBasisFuncsAll(degree,knot_vector,params)
    result=np.linalg.lstsq(basis,points,rcond=None)
    solution=result[0]
    residual=basis@solution-points
    return solution,residual

def approximateCurve(degree,knot_vector,points,params):
    '''
        return ctrl_pts,loss
    '''
    basis=getBasisFuncsAll(degree,knot_vector,params)
    ctrl_pts= np.linalg.lstsq(basis,points,rcond=None)[0]
    return BSplineCurve(degree,ctrl_pts,knot_vector)

def ErrorFLossSingle(curve:BSplineCurve,point,param):
    '''
        from paper Eric Saux, An improved Hoschek intrinsic parametrization, 2003
        return gradient of error function
    '''
    point=np.array(point)
    # param=np.clip(param,0,1)
    if param<1e-8 or param>1-1e-8 or np.isnan(param):
        return 10000


    if type(curve)==BSpline.Curve:
        crv=curve
    else:
        crv=BSpline.Curve()
        crv.degree=curve.degree
        crv.ctrlpts=curve.control_pts.tolist()
        crv.knotvector=curve.knot_vector

    # ders=[crv.derivatives(u,order=1) for u in params]
    try:
        coordinates=crv.evaluate_single(param)
    except GeomdlException:
        print ('error param:',param)
        raise

    coordinates=np.array(coordinates)
    diff=coordinates-point
    
    loss=np.linalg.norm(diff,axis=-1)
    
    return loss

def ErrorFLoss(curve:BSplineCurve,points,params):
    '''
        from paper Eric Saux, An improved Hoschek intrinsic parametrization, 2003
        return gradient of error function
    '''
    points=np.array(points)
    params=np.clip(params,0,1)

    if type(curve)==BSpline.Curve:
        crv=curve
    else:
        crv=BSpline.Curve()
        crv.degree=curve.degree
        crv.ctrlpts=curve.control_pts.tolist()
        crv.knotvector=curve.knot_vector

    # ders=[crv.derivatives(u,order=1) for u in params]
    coordinates=crv.evaluate_list(param_list=params)
    coordinates=np.array(coordinates)
    diff=coordinates-points
    
    loss=np.linalg.norm(diff,axis=-1)
    
    return loss

def ErrorFGradient(curve:BSplineCurve,points,params):
    '''
        from paper Eric Saux, An improved Hoschek intrinsic parametrization, 2003
        return gradient of error function
    '''
    points=np.array(points)
    # params=np.array(params).clip(0,1)

    if type(curve)==BSpline.Curve:
        crv=curve
    else:
        crv=BSpline.Curve()
        crv.degree=curve.degree
        crv.ctrlpts=curve.control_pts.tolist()
        crv.knotvector=curve.knot_vector

    ders=[crv.derivatives(u,order=1) for u in params]
    ders=np.array(ders)
    coordinates=ders[:,0,:]
    ders=ders[:,1,:]
    diff=coordinates-points
    
    loss=(np.sum(diff*ders,axis=-1))/(np.linalg.norm(diff,axis=-1))
    
    return loss
def reparameterizeCurve(curve:BSplineCurve,points,params,max_it=1000):
    '''
        return new curves with new params
    '''
    crv=BSpline.Curve()
    crv.degree=curve.degree
    crv.ctrlpts=curve.control_pts.tolist()
    crv.knotvector=curve.knot_vector
    
    params_k=params.clip(0,1)
    g_loss_k=ErrorFGradient(crv,points,params_k)
    dk=-g_loss_k
    # convergence=False
    tol=1e-7
    
    # convergence_flags=np.zeros_like(params_k,dtype=np.bool_)
    convergence_flags=np.zeros_like(params_k)
    
    if max_it is not None:
        count_down=max_it
    else:
        count_down=0
    
    while not np.all(convergence_flags):
        # params_k=params_k+dk
        params_k_1=params_k

        assert not np.any(np.isnan(dk))
        lmda=[brent(lambda x:ErrorFLossSingle(curve,point,param+x*dk_single),tol=1e-6) 
                  if not skip else 0 for point,param,dk_single,skip in zip(points,params_k,dk,convergence_flags)]
        lmda=np.array(lmda)
        params_k=params_k_1+lmda*dk

        # params_k=[brent(lambda x:ErrorFLossSingle(curve,point,param+x*dk_single),brack=(0,1e-4)) 
        #           for point,param,dk_single,skip in zip(points,params_k,dk,convergence_flags)]

        params_k=np.array(params_k).clip(1e-8,1-1e-8)
        
        dk_1=dk 
        g_loss_k_1=g_loss_k
        g_loss_k=ErrorFGradient(crv,points,params_k)
        beta=(g_loss_k/g_loss_k_1)**2
        dk=-g_loss_k+beta*dk_1
        
        convergence_flags=np.abs(params_k-params_k_1)<tol*(10*np.abs(params_k_1)+1)
        # convergence=np.all(np.abs(params_k-params_k_1)<tol*(10*np.abs(params_k_1)+1))
        
        count_down-=1
        if count_down==0:
            break

    params_k[0]=0
    params_k[-1]=1
    return params_k

def testBasis():
    gen=BSplineCurveGenerator()
    curve=gen.genCurves(1,ctrl_number=7)[0]
    knots=np.linspace(0,1,10)
    print(getBasisFuncsAllFromCurve(curve,knots))

def testApproximation():
    gen=BSplineCurveGenerator()
    curve=gen.genCurves(1,ctrl_number=7)[0]
    points,knots=samplePointsFromCurve(curve,20,return_params=True)
    # print(getBasisFuncsAllFromCurve(curve,knots))
    print(approximateCtrlPts(curve.degree,curve.knot_vector,points,knots))
    print('label:')
    print(curve.control_pts)
if __name__=='__main__':
    # testBasis()
    testApproximation()

