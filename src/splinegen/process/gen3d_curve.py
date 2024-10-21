import os.path as path
import os
import sys
from tqdm import tqdm

fdir = path.dirname
sys.path.append(fdir(fdir(path.abspath(__file__))))

import util.curve as curve
import numpy as np

def gen_curve_all3d(save_path,num_base=10000,insert_times=4,num_noise_layers=10,num_sample_times=1,
                     noise=0.4,
                     scale=64,
                     dimension=3,
                     degree=3,
                    #    paramter_generator=curve.EqualChordLengthParameterGenerator,
                     num_ctrl_pts_min=4,num_ctrl_pts_max=8,min_samples=50,max_samples=100,
                     ctrl_pts_generator=curve.MovingCenterControlPointsGenerator):
    '''
        Corresponding to dataset/curveDataset2.py
    '''
    parent_path=fdir(save_path)
    if parent_path and not path.exists(parent_path):
        os.makedirs(parent_path)

    generator=curve.BSplineCurveGenerator(degree=degree,dimension=dimension,knot_vector_generator=curve.UniformNonPeriodicKnotVectorGenerator,
                                                ctrl_num_dist='uniform',
                                               max_ctrl_num=num_ctrl_pts_max,min_ctrl_num=num_ctrl_pts_min,control_pts_generator=ctrl_pts_generator) 
                               
    eps=1e-2

    from util.curveIntersect2D import isSelfIntersect3D
    def isIntersect(c):
        return isSelfIntersect3D(c,eps=eps,scale=scale)

    print('gen base curves...')
    base_curves=generator.genNormalizedCurves(num=num_base,use_tqdm=True)
    base_curves=[c for c in tqdm(base_curves) if not isIntersect(c)]
    
    num_base=len(base_curves)

    for i in range(insert_times):
        print(f'insert knot {i+1}/{insert_times} times...')
        for j in tqdm(range(num_base)):
            # base_curves.append(curve.disturbCurveKnotsMulti(base_curves[i*num_base+j],interior_disturb=0.7))
            new_c=curve.knotInsertionAtLargestSpanMulti(base_curves[i*num_base+j],times=num_ctrl_pts_max-num_ctrl_pts_min)
            # new_c=curve.disturbCurveKnotsMulti(new_c,interior_disturb=0.1).getNormalized()
            base_curves.append(new_c)
    
    # print('checking self intersecting...')
    # base_curves=[c for c in tqdm(base_curves) if not isIntersect(c)]

    CTRL_NUM_MAX_LIMIT=100
    ctrl_num_bin=np.zeros(CTRL_NUM_MAX_LIMIT,dtype=np.int64)
    curves=[]

    max_frequency=0

    for c in base_curves:
        ctrl_num=len(c.control_pts)
        num_ctrl_pts_max=max(num_ctrl_pts_max,ctrl_num)
        num_ctrl_pts_min=min(num_ctrl_pts_min,ctrl_num)
        ctrl_num_bin[ctrl_num]+=1 
        
        if ctrl_num_bin[ctrl_num]>max_frequency:
            max_frequency=ctrl_num_bin[ctrl_num]

    print(num_ctrl_pts_max,num_ctrl_pts_min)

    print('add noise...')
    for c in tqdm(base_curves):
        ctrl_num=len(c.control_pts)
        times=num_noise_layers*max_frequency/ctrl_num_bin[ctrl_num]
        # times=num_noise_layers
        
        def add_noise(state={'noise_scale':noise,'init':True},decay=1,max_try=100):
            if state['init']:
                origin=c
            else:
                origin=curves[-1]

            new_c=curve.disturbCurveKnotsMulti(origin,interior_disturb=state['noise_scale'])
            
            cnt=0
            while isIntersect(new_c):
                state['noise_scale']*=decay
                new_c=curve.disturbCurveKnotsMulti(origin,interior_disturb=state['noise_scale'])
                cnt+=1
                if cnt>max_try:
                    print(f'reach limit: ctrl num {ctrl_num}')
                    raise TimeoutError()
                
            state['init']=False
            curves.append(new_c)
            
        try:
            while times>=1:
                add_noise()
                times-=1
            if np.random.rand()<times:
                add_noise()
        except TimeoutError:
            continue

    print('make regular arrays...')
    ctrl_pts=np.zeros((len(curves),num_ctrl_pts_max,dimension)) 
    ctrl_pts_len=np.zeros((len(curves)),dtype=np.int64) 
    knots=np.zeros((len(curves),num_ctrl_pts_max+degree+1)) 
    
    for id,c in enumerate(tqdm(curves)):
        c_len=len(c.control_pts)
        ctrl_pts_len[id]=c_len
        ctrl_pts[id,:c_len]=c.control_pts
        knots[id,:c_len+c.degree+1]=c.knot_vector
        
    print(f'sample points {num_sample_times} times...')
    points_array=np.zeros((num_sample_times,len(curves),max_samples,dimension))
    params_array=np.zeros((num_sample_times,len(curves),max_samples))
    points_len=np.zeros((num_sample_times,len(curves)))

    from util.curveSampling2D import NoisedEqualChordLenSamplingParamsCpp3D

    for i in range(num_sample_times):
        print(f'sample points batch {i+1}/{num_sample_times}:')
        points,params=curve.samplePointsFromCurves(
            curves=curves,
            max_point_length=max_samples,
            return_params=True,
            min_point_length=min_samples,
            fix_size=False,
            use_tqdm=True,
            parameter_generator=NoisedEqualChordLenSamplingParamsCpp3D)
        
        print('make regular arrays...')
        for j in tqdm(range(len(curves))):
            len_p=len(points[j])
            points_array[i,j,:len_p] = points[j]
            params_array[i,j,:len_p] = params[j]
            points_len[i,j]=len_p
        
    ctrl_pts_array=np.broadcast_to(np.expand_dims(ctrl_pts,axis=0),(num_sample_times,*ctrl_pts.shape)).reshape(-1,*ctrl_pts.shape[1:])
    ctrl_pts_len=np.broadcast_to(np.expand_dims(ctrl_pts_len,axis=0),(num_sample_times,*ctrl_pts_len.shape)).reshape(-1)
    knots_array=np.broadcast_to(np.expand_dims(knots,axis=0),(num_sample_times,*knots.shape)).reshape(-1,*knots.shape[1:])
    points_array=np.reshape(points_array,(-1,*points_array.shape[2:]))
    params_array=np.reshape(params_array,(-1,*params_array.shape[2:]))
    points_len=np.reshape(points_len,(-1))
    
    print(f'saving to {save_path}...')
    np.savez(save_path,ctrl_pts=ctrl_pts_array,
             ctrl_pts_len=ctrl_pts_len,knots=knots_array,
             points=points_array,params=params_array,points_len=points_len,degree=degree)
    
    print(f'done: total {len(points_len)} curves')


def gen3d():
    save_path=f'3d_curve_50w.npz'
    gen_curve_all3d(save_path=save_path,num_base=6000,num_noise_layers=10,num_sample_times=1,num_ctrl_pts_min=4,num_ctrl_pts_max=8,insert_times=4,min_samples=30,max_samples=50,scale=64)


if __name__=='__main__':
    np.random.seed(23)
    gen3d()