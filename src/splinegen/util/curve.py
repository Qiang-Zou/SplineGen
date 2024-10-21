from typing import List,Callable
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from geomdl import BSpline,NURBS
import json

from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array1OfPnt2d as Pnt2d
from OCC.Core.TColStd import TColStd_Array1OfReal as RealArray, TColStd_Array1OfInteger as IntArray
from OCC.Core.Geom2d import Geom2d_BSplineCurve
from OCC.Core.Geom2dAPI import Geom2dAPI_InterCurveCurve
from tqdm import tqdm

class BSplineCurve:
    def __init__(self,degree,control_pts:List,knot_vector:List[float]) -> None:
        '''
            Args: 
                control_pts: [num_ctrl_pts,dimension]
                knot_vector: [num_ctrl_pts+degree+1]
        '''
        self.degree=degree
        self.control_pts=np.array(control_pts)
        self.dimension=np.shape(control_pts)[-1]
        self.knot_vector=knot_vector
        
        # assert len(control_pts)==2
        # assert len(knot_vector)==1
        # assert len(control_pts)+degree+1==len(knot_vector)
    def getNormalized(self,rotate=True,diag=True,eps=1e-2,new_center=0):
        c=np.array(self.control_pts)
        c=c-np.mean(c,axis=0)
        if rotate:
            c=PCA(n_components=self.dimension).fit_transform(c)
        
        if diag:
            if self.dimension==2:
                c= np.array([[1,-1],[1,1]]).dot(c.T).T
            elif self.dimension==3:
                c= np.array([[1,0,0],[0,1,-1],[0,1,1]]).dot(np.array([[1,-1,0],[1,1,0],[0,0,1]]).dot(c.T)).T

        center=(np.max(c,axis=0)+np.min(c,axis=0))/2
        span=np.max(c,axis=0)-np.min(c,axis=0)
        span=np.max(span)
        c = (c-center)/(span+eps)
        c=c+new_center
        return BSplineCurve(self.degree,c,self.knot_vector)
    
    def rotate(self,rotate_matrix):
        # c= self.control_pts.dot(rotate_matrix)
        c=rotate_matrix.dot(c.T).T
        return BSplineCurve(self.degree,c,self.knot_vector)
    def rotateByAngle(self,angle):
        rotate_matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        c=rotate_matrix.dot(self.control_pts.T).T 
        return BSplineCurve(self.degree,c,self.knot_vector)

    def rotateByAngleZ(self,angle):
        rotate_matrix=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        c=rotate_matrix.dot(self.control_pts.T).T 
        return BSplineCurve(self.degree,c,self.knot_vector)
    def rotateByAngleY(self,angle):
        rotate_matrix=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        c=rotate_matrix.dot(self.control_pts.T).T 
        return BSplineCurve(self.degree,c,self.knot_vector)
    def rotateByAngleX(self,angle):
        rotate_matrix=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        c=rotate_matrix.dot(self.control_pts.T).T 
        return BSplineCurve(self.degree,c,self.knot_vector)
    
    def isKnotsTooClose(self,eps=1e-4)->bool:
        knots=[]
        multis=[]
        for knot in self.knot_vector:
            if len(knots) >0 and (knot-knots[-1])<eps:
                multis[-1]+=1
            else:
                knots.append(knot)
                multis.append(1)
                
            if knot!=0 and knot!=1 and multis[-1]>1:
                return True
        return False

    def isSelfIntersecting(self,restrict=True,eps=1e-4)->bool:
        if self.dimension!=2 and self.dimension!=3:
            print('only support 2d curve now')
            raise NotImplementedError
        knots=[]
        multis=[]
        for knot in self.knot_vector:
            if len(knots) >0 and (knot-knots[-1])<eps:
                multis[-1]+=1
            else:
                knots.append(knot)
                multis.append(1)
                
            if knot!=0 and knot!=1:
                if restrict and multis[-1]>1:
                    return True
                
                if multis[-1]>self.degree:
                    return True

        if self.dimension==3:
            return False
        knots_=RealArray(1,len(knots))
        multis_=IntArray(1,len(multis))
        ctrls=Pnt2d(1,len(self.control_pts))

        for i in range(len(knots)):
            knots_[i]=knots[i]
        for i in range(len(multis)):
            multis_[i]=multis[i]
            
        for i in range(len(self.control_pts)):
            pt=gp_Pnt2d(self.control_pts[i,0],self.control_pts[i,1])
            ctrls[i]=pt
            
        try:
            curve=Geom2d_BSplineCurve(ctrls,knots_,multis_,self.degree)
            intersector=Geom2dAPI_InterCurveCurve(curve)
        except:
            return True
        
        return intersector.NbPoints()>0
        
    def evaluate(self,t):
        crv=BSpline.Curve()
        crv.degree=self.degree
        crv.ctrlpts=self.control_pts.tolist()
        crv.knotvector=self.knot_vector
        
        point=crv.evaluate_single(t)
        
        return point

    def __repr__(self) -> str:
        return f'degree:{self.degree}\ncontrol_pts:{self.control_pts}\nknot_vector:{self.knot_vector}'
    
def knotInsertion(curve:BSplineCurve,t,m=1):
    knots=curve.knot_vector
    ctrl_pts=curve.control_pts
    degree=curve.degree
    
    crv=BSpline.Curve()
    crv.degree=degree
    crv.ctrlpts=ctrl_pts.tolist()
    crv.knotvector=knots
    
    crv.insert_knot(param=t,num=m) 
    
    return BSplineCurve(degree=degree,control_pts=crv.ctrlpts,knot_vector=crv.knotvector)


def knotInsertionAtLargestSpan(curve:BSplineCurve):
    knots=np.array(curve.knot_vector)
    spans=knots[1:]-knots[:-1]
    max_span_index=np.argmax(spans)
    new_knot=(knots[max_span_index]+knots[max_span_index+1])/2
    return knotInsertion(curve=curve,t=new_knot)

def knotInsertionAtLargestSpanMulti(curve:BSplineCurve,times):
    for _ in range(times):
        curve=knotInsertionAtLargestSpan(curve)
    return curve

def UniformControlPointsGenerator(ctrl_num,curve_num=1,dimension=2):
    return np.random.rand(curve_num,ctrl_num,dimension)

def MovingCenterControlPointsGenerator(ctrl_num,curve_num=1,dimension=2):
    if dimension==3:
        points_x=[]
        points_y=[]
        points_z=[]
        for i in range(ctrl_num):
            x=np.random.normal(10+i,1,size=curve_num)
            y=np.random.normal(10+i,2,size=curve_num)
            z=np.random.normal(0,1,size=curve_num)
            points_x.append(x)
            points_y.append(y)
            points_z.append(z)
        points_x=np.array(points_x)
        points_y=np.array(points_y)
        points_z=np.array(points_z)
        return np.stack([points_x,points_y,points_z],axis=-1).transpose(1,0,2)
    if dimension!=2:
        raise NotImplementedError
    points_x=[]
    points_y=[]
    for i in range(ctrl_num):
        x=np.random.normal(10+i,1,size=curve_num)
        y=np.random.normal(10+i,2,size=curve_num)
        points_x.append(x)
        points_y.append(y)
    points_x=np.array(points_x)
    points_y=np.array(points_y)
    return np.stack([points_x,points_y],axis=-1).transpose(1,0,2)


def UniformParameterGenerator(curve:BSplineCurve,curve_num=1,size=1000):
    params= np.random.rand(curve_num,size)
    params[:,0]=0
    params[:,-1]=1
    return params

def parameterAtLength(length,origin_params,curve:BSpline.Curve,chord:List[float],eps=1e-7):
    # t = 0.0
    if length <= chord[0]+ eps:
        return 0.0
    elif length >= chord[-1]- eps:
        return 1.0
    else:
        low = 0
        high =len(chord)-1

        while high - low > 1:
            mid = (high + low) >> 1

            if (length < chord[mid] - eps):
                high = mid
            else:
                low = mid
        t0 = origin_params[low]
        t1 = origin_params[low+1]
        p0 = curve.evaluate(t0)
        p1 = curve.evaluate(t1)
        remainingLength = length - chord[low]
        
        segmentLength = np.linalg.norm(np.array(p1)-np.array(p0))

        alpha = remainingLength / segmentLength
        t = t0 + alpha * (t1 - t0)
        return t

def EqualChordLengthParameterGenerator(curve:BSplineCurve,curve_num=1,size=1000,revolution=1000):
    crv=BSpline.Curve()
    crv.degree=curve.degree
    crv.ctrlpts=curve.control_pts.tolist()
    crv.knotvector=curve.knot_vector
    
    param_list=np.linspace(0,1,num=revolution)
    points=crv.evaluate_list(param_list)
    points=np.array(points)
    
    distance=points[1:]-points[:-1]
    distance=np.linalg.norm(distance,axis=-1)
    chord=np.cumsum(distance)
    
    numSamples=size

    totalLength = chord[-1]
    segmentLength = totalLength / (numSamples-1)

    new_params=[]
    for i in range(numSamples):
        targetLength = i * segmentLength
        t = parameterAtLength(targetLength,param_list,curve,chord)
        
        new_params.append(t)
        
    return [new_params]


def ConstantStepParameterGenerator(curve:BSplineCurve,size=1000):
    return [np.linspace(0,1,num=size)]

def UniformNonPeriodicKnotVectorGenerator(ctrl_num,degree,curve_num=1):
    interior=np.random.rand(curve_num,ctrl_num-degree+1)
    interior=np.cumsum(interior,axis=1)
    max_interior=interior[:,-1]
    min_interior=interior[:,0]
    interior-=min_interior
    interior/=max_interior-min_interior

    vectors=np.zeros((curve_num,degree+1+ctrl_num),dtype=np.float64)
    vectors[:,-degree:]=1.
    vectors[:,degree:-degree]=interior
    return vectors

def ConstantStepNonPeriodicKnotVectorGenerator(ctrl_num,degree,curve_num=1):
    interior=np.linspace(0,1,num=ctrl_num-degree+1)
    interior=np.tile(interior,(curve_num,1))

    vectors=np.zeros((curve_num,degree+1+ctrl_num),dtype=np.float64)
    vectors[:,-degree:]=1.
    vectors[:,degree:-degree]=interior
    return vectors

def NoInteriorKnotVectorGenerator(ctrl_num,degree,curve_num=1):
    assert ctrl_num>=degree+1
    vectors=np.zeros((curve_num,2*degree+2),dtype=np.float64)
    vectors[:,degree+1:]=1.
    return vectors

def disturbCurveKnots_(curve:BSplineCurve,interior_disturb=0.1):
    noise=np.random.normal(loc=0,scale=interior_disturb,size=len(curve.knot_vector)-2*(curve.degree+1))
    curve.knot_vector[curve.degree+1:-curve.degree-1]+=noise
    curve.knot_vector.sort()

def disturbCurveKnotsMulti(curve:BSplineCurve,interior_disturb=0.1,eps=1e-3,times=1):
    noise=np.random.normal(loc=0,scale=interior_disturb,size=(len(curve.knot_vector)-2*(curve.degree+1),times))
    noise=np.sum(noise,axis=-1)
    knots=np.array(curve.knot_vector)
    new_interior=knots[curve.degree+1:-curve.degree-1]+noise
    new_interior=np.clip(new_interior,eps,1-eps)
    new_interior=np.sort(new_interior)
    knots[curve.degree+1:-curve.degree-1]=new_interior
    
    return BSplineCurve(degree=curve.degree,control_pts=curve.control_pts,knot_vector=knots)

def disturbCurveKnots(curve:BSplineCurve,interior_disturb=0.1,eps=1e-3):
    noise=np.random.normal(loc=0,scale=interior_disturb,size=len(curve.knot_vector)-2*(curve.degree+1))
    knots=np.array(curve.knot_vector)
    new_interior=knots[curve.degree+1:-curve.degree-1]+noise
    new_interior=np.clip(new_interior,eps,1-eps)
    new_interior=np.sort(new_interior)
    knots[curve.degree+1:-curve.degree-1]=new_interior
    
    return BSplineCurve(degree=curve.degree,control_pts=curve.control_pts,knot_vector=knots)

class BSplineCurveGenerator:
    def __init__(self,
                 degree=3,
                 dimension=2,
                 max_ctrl_num=10,
                 min_ctrl_num=None,
                 control_pts_generator:Callable[[int,int,int],List]=UniformControlPointsGenerator,
                 knot_vector_generator:Callable[[int,int,int],List[float]]=UniformNonPeriodicKnotVectorGenerator,
                 parameter_generator:Callable[[int,int,int],List[float]]=UniformParameterGenerator,
                 ctrl_num_dist='uniform'
                 ) -> None:
        self.degree=degree
        self.dimension=dimension
        self.max_ctrl_num=max_ctrl_num
        self.c_gen=control_pts_generator
        self.k_gen=knot_vector_generator
        self.p_gen=parameter_generator
        
        if min_ctrl_num is None:
            self.min_ctrl_num=degree+1
        else:
            self.min_ctrl_num=min_ctrl_num
            
        self.ctrl_num_dist=ctrl_num_dist
    
    def genCurves(self,num:int,ctrl_number=None)->List[BSplineCurve]:
        if ctrl_number is not None:
            return [BSplineCurve(self.degree,self.c_gen(ctrl_number,dimension=self.dimension)[0],self.k_gen(ctrl_number,self.degree)[0]) 
                    for _ in range(num)]

        ctrl_num_dist=self.ctrl_num_dist 
        if ctrl_num_dist=='uniform':
            it=np.random.randint(self.min_ctrl_num,self.max_ctrl_num+1,num)
        elif ctrl_num_dist=='normal':
            it=map(int,np.random.normal(self.max_ctrl_num,8,num))
        elif ctrl_num_dist=='exp':
            it=map(int,self.max_ctrl_num-np.random.exponential(1,num))

        return [BSplineCurve(self.degree,self.c_gen(ctrl_num,dimension=self.dimension)[0],self.k_gen(ctrl_num,self.degree)[0]) 
                for ctrl_num in it if ctrl_num>=self.min_ctrl_num and ctrl_num<=self.max_ctrl_num]
        
    def genNormalizedCurves(self,num:int,ctrl_number=None,use_tqdm=False)->List[BSplineCurve]:
        if use_tqdm:
            return [curve.getNormalized() for curve in tqdm(self.genCurves(num=num,ctrl_number=ctrl_number))]
        return [curve.getNormalized() for curve in self.genCurves(num=num,ctrl_number=ctrl_number)]
    
    def genNonSelfIntersectingCurves(self,num:int,ctrl_number=None,use_tqdm=False)->List[BSplineCurve]:
        if use_tqdm:
            return [curve for curve in tqdm(self.genNormalizedCurves(num=num,ctrl_number=ctrl_number)) if not curve.isSelfIntersecting()]
        return [curve for curve in self.genNormalizedCurves(num=num,ctrl_number=ctrl_number) if not curve.isSelfIntersecting()]
    

def getNormalizedCurves(curves:List[BSplineCurve]):
    pass

def refineSamplingFollowingKnotsDist(params,knots,degree):
    intervals=len(knots)-2*degree-1
    params_scale=params*intervals
    low_kont_index=np.floor(params_scale).astype(np.int64)
    offset=params_scale-low_kont_index
    knot_base=knots[degree+low_kont_index]
    knot_ceil=knots[degree+low_kont_index+1]
    
    return (1-offset)*knot_base+offset*knot_ceil

def samplePointsFromCurve(curve:BSplineCurve,max_point_length=100,min_point_length=50,fix_size=True,given_list=None,
                          parameter_generator=UniformParameterGenerator,return_params=False,follow_knots_dist=False)->List[List[float]]:
    crv=BSpline.Curve()
    crv.degree=curve.degree
    crv.ctrlpts=curve.control_pts.tolist()
    crv.knotvector=curve.knot_vector
    
    if given_list is not None:
        param_list_=given_list
    else:
    
        if fix_size:
            param_list_=parameter_generator(curve,size=max_point_length)[0]
        else:
            param_list_=parameter_generator(curve,size=np.random.randint(min_point_length,max_point_length+1))[0]
            
    if follow_knots_dist:
        param_list_=refineSamplingFollowingKnotsDist(param_list_,curve.knot_vector,curve.degree)
    
    param_list=np.sort(param_list_)
    param_list=np.clip(param_list,0,1) 
    # param_list=np.sorted(param_list_)

    # points=crv.evaluate_list(param_list)
    points=[crv.evaluate_single(param) for param in param_list]
    if return_params:
        return points,param_list
    return points


def samplePointsFromCurves(curves:List[BSplineCurve],max_point_length=100,min_point_length=50,fix_size=False,
                           return_params=False,
                           numpy=True,use_tqdm=False,parameter_generator=UniformParameterGenerator):
    it=curves
    if use_tqdm:
        it=tqdm(curves)
        
    if not return_params:
        if numpy:
            return [np.array(samplePointsFromCurve(curve,max_point_length=max_point_length,
                                          fix_size=fix_size,parameter_generator=parameter_generator,min_point_length=min_point_length)) for curve in it]
            
        return [samplePointsFromCurve(curve,max_point_length=max_point_length,
                                      fix_size=fix_size,parameter_generator=parameter_generator,min_point_length=min_point_length) for curve in it]
        
    points=[]
    params=[]
    for curve in it:
        p,p_list=samplePointsFromCurve(curve,max_point_length=max_point_length,
                                      fix_size=fix_size,parameter_generator=parameter_generator,min_point_length=min_point_length,return_params=True)
        if numpy:
            points.append(np.array(p))
            params.append(np.array(p_list))
        else:
            points.append(p)
            params.append(p_list)

    return points,params

def drawBSplineCurvesCtrlPts(curves:List[BSplineCurve]):
    pass

def drawBSplineCurveCtrlPts(curve:BSplineCurve,draw_ctrl_label=True,fig=None,ax=None,**kwargs):
    if ax is None:
        fig,ax=plt.subplots()
    ax.scatter(curve.control_pts[:,0],curve.control_pts[:,1],**kwargs)
    if draw_ctrl_label:
        for i in range(len(curve.control_pts)):
            ax.text(curve.control_pts[i,0],curve.control_pts[i,1],f'P{i}')
    return fig,ax

def drawBSplineCurveCtrlPtsLine(curve:BSplineCurve,fig=None,ax=None,**kwargs):
    if ax is None:
        fig,ax=plt.subplots()
    ax.scatter(curve.control_pts[:,0],curve.control_pts[:,1],**kwargs)
    ax.plot(curve.control_pts[:,0],curve.control_pts[:,1],**kwargs)
    return fig,ax

def drawBSplineCurves(curves:List[BSplineCurve]):
    pass

def drawBSplineCurve(curve:BSplineCurve,
                     sample_size=64,
                     draw_ctrls=True,
                     draw_ctrl_label=True,
                     draw_ctrl_lines=True,
                     fig=None,ax=None,
                     line_color='b',
                     ctrl_pts_color='r',
                     ctrl_lines_color='k',
                     label='curve',
                     parameter_generator=EqualChordLengthParameterGenerator):
    points=samplePointsFromCurve(curve,max_point_length=sample_size,parameter_generator=parameter_generator,follow_knots_dist=False)
    points=np.array(points)
    if ax is None:
        fig,ax=plt.subplots()
    ax.plot(points[:,0],points[:,1],c=line_color,label=label)
    if draw_ctrls:
        drawBSplineCurveCtrlPts(curve,ax=ax,fig=fig,draw_ctrl_label=draw_ctrl_label,c=ctrl_pts_color)
    if draw_ctrl_lines:
        drawBSplineCurveCtrlPtsLine(curve,ax=ax,fig=fig,c=ctrl_lines_color)
    return fig,ax

def drawBSplineCurve3DStem(curve:BSplineCurve,sample_size=64):
    points=samplePointsFromCurve(curve,max_point_length=sample_size,parameter_generator=ConstantStepParameterGenerator)
    points=np.array(points)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    markerline, stemlines, baseline = ax.stem(points[:,0],points[:,1],points[:,2],markerfmt='D',linefmt='grey')
    markerline.set_markerfacecolor('none')
    # drawBSplineCurveCtrlPts(curve,c='r')
    # drawBSplineCurveCtrlPtsLine(curve,c='k')
    return fig,ax

def drawBSplineCurve3D(curve:BSplineCurve,sample_size=64,show_ctrl_points=False,scatter=False,view='xyz',title=None,label='Curve',
                        fig=None,ax=None,parameter_generator=ConstantStepParameterGenerator
                       ):
    points=samplePointsFromCurve(curve,max_point_length=sample_size,parameter_generator=parameter_generator)
    points=np.array(points)

    if fig is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    
    view_dict={'x':0,'y':1,'z':2}
    x,y,z=view_dict[view[0]],view_dict[view[1]],view_dict[view[2]]
    
    if scatter:
        ax.scatter(points[:,x],points[:,y],points[:,z],c='b',label=label)
    else:
        ax.plot(points[:,x],points[:,y],points[:,z],c='b',label=label)

    if show_ctrl_points:
        ax.scatter(curve.control_pts[:,x],curve.control_pts[:,y],curve.control_pts[:,z],c='r')
        ax.plot(curve.control_pts[:,x],curve.control_pts[:,y],curve.control_pts[:,z],c='k')
    # drawBSplineCurveCtrlPts(curve,c='r')
    # drawBSplineCurveCtrlPtsLine(curve,c='k')
    if title is not None:
        ax.set_title(title)
    return fig,ax

def saveCurves(location:str,curves:List[BSplineCurve],format='npz'):
    if format=='npz':
        np.savez(location,curves=curves)
    elif format=='json':
        dictionary={
            'degree':curves[0].degree,
            'control_pts':[curve.control_pts.tolist() for curve in curves],
            'knot_vectors':[curve.knot_vector.tolist() for curve in curves]
        }
        with open(location, "w") as outfile:
            json.dump(dictionary, outfile)

def loadCurves(location:str,format='npz')->List[BSplineCurve]:
    if format=='npz':
        return np.load(location)['curves']        
    elif format=='json':
        with open(location, 'r') as openfile:
            json_object = json.load(openfile)
        return [BSplineCurve(json_object['degree'],np.array(json_object['control_pts'][i]),
                             np.array(json_object['knot_vectors'][i])) for i in range(len(json_object['control_pts']))]
                             
    elif format=='npz2':
        # corresponding to curveDataset2
        
        print('data loading...')
        data=np.load(location)

        degree=data['degree']

        print('degree:',degree)

        ctrl_pts=data['ctrl_pts']
        _,max_ctrl_len,__=ctrl_pts.shape
        print('ctrl_pts shape:',ctrl_pts.shape)
        ctrl_pts_len_array=data['ctrl_pts_len'] # (num_curve)

        ctrl_pts=data['ctrl_pts']
        
        knots=data['knots']
        
        curves=[]
        for c,cl,k in tqdm(zip(ctrl_pts,ctrl_pts_len_array,knots)):
            ctrl=c[:cl]
            knots_vector=k[:(cl+degree+1)]
            curves.append(BSplineCurve(degree=int(degree),control_pts=ctrl,knot_vector=knots_vector))
        return curves
            

        

###### TEST ########### 

def save_test():
    generator=BSplineCurveGenerator(dimension=3,max_ctrl_num=12)
    curves=generator.genCurves(10000)
    saveCurves('data/curves.npz',curves=curves)
def save_test2():
    generator=BSplineCurveGenerator(dimension=3,max_ctrl_num=12)
    curves=generator.genCurves(10000)
    saveCurves('data/curves.json',curves=curves,format='json')
def save_load_test():
    generator=BSplineCurveGenerator(dimension=3,max_ctrl_num=12)
    curves=generator.genCurves(2)
    print(curves[0].control_pts)
    print(curves[1].control_pts)
    print('#######################')

    saveCurves('data/curves.json',curves=curves,format='json')
    
    curves=loadCurves('data/curves.json',format='json')
    print(curves[0].control_pts)
    print(curves[1].control_pts)
    
def test_intersecting():
    generator=BSplineCurveGenerator(dimension=2,max_ctrl_num=12)
    curves=generator.genCurves(num=1,ctrl_number=12)
    
    isIntersect=curves[0].isSelfIntersecting()
    
    print(isIntersect)
    
def test_disturb():
    generator=BSplineCurveGenerator(dimension=2,max_ctrl_num=12)
    curves=generator.genCurves(num=1,ctrl_number=12)
    
    print(curves[0].knot_vector)
    
    c=disturbCurveKnots(curves[0],0.01)
    
    print(c.knot_vector)
    
def testRefineSampling():
    generator=BSplineCurveGenerator(dimension=2,max_ctrl_num=12)
    curves=generator.genCurves(num=1)
    crv=curves[0] 
    print(crv.knot_vector)
    p=UniformParameterGenerator(1,20)
    p=np.sort(p)
    print(p)
    p=refineSamplingFollowingKnotsDist(params=p,knots=crv.knot_vector,degree=crv.degree)
    print(p)

def main():
    # test_disturb()
    testRefineSampling()

if __name__ == "__main__":
    main()