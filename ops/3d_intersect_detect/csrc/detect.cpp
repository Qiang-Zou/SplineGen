#include <cstddef>
#include <algorithm>
#include <random>
#include<cstdio>
#include<iostream>
#include<cmath>
#include <ostream>
#include<vector>
#include<fstream>
#include "detect.hpp"

const int maxn = 1e3 + 5;

bool judge[maxn][maxn][maxn];

#define eps 1e-7
#define pb push_back

struct Point3D 
{
    double x;
    double y;
    double z;
};

inline double distance(const Point3D& p1, const Point3D& p2)
{
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}


class BSplineCurve3D
{
public:
    explicit BSplineCurve3D(const int &degree, const std::vector<Point3D>& controlPoints, const std::vector<double>& knots) : degree_(degree), controlPoints_(controlPoints), knots_(knots)
    {
        n_ = controlPoints.size() - 1;
    }

    inline Point3D evaluate(double t) const
    {
        int span = findSpan(t);
        std::vector<double> basis = computeBasis(span, t);
        Point3D result = { 0.0, 0.0, 0.0 };

        for (size_t i = 0; i <= degree_; ++i)
        {
            double basisMultiplier = basis[i];
            result.x += controlPoints_[span - degree_ + i].x * basisMultiplier;
            result.y += controlPoints_[span - degree_ + i].y * basisMultiplier;
            result.z += controlPoints_[span - degree_ + i].z * basisMultiplier;
        }

        return result;
    }
    void dump_curve()const{
        std::cout<<"control points: "<<std::endl;
        for (size_t i = 0; i < controlPoints_.size(); i++)
        {
            std::cout<<"("<<controlPoints_[i].x<<", "<<controlPoints_[i].y<<", "<<controlPoints_[i].z<<")"<<std::endl;
        }
        std::cout<<"knots: "<<std::endl;
        for (size_t i = 0; i < knots_.size(); i++)
        {
            std::cout<<knots_[i]<<std::endl;
        }
        std::cout<<"degree: "<<degree_<<std::endl;
    }

private:
    inline int findSpan(double t) const
    {
        int low = degree_;
        int high = n_ + 1;
        int mid;

        while (high - low > 1)
        {
            mid = (high + low) >> 1;

            if (t < knots_[mid]-eps)
                high = mid;
            else
                low = mid;
        }

        return low;
    }

    inline std::vector<double> computeBasis(int span, double t) const
    {
        std::vector<double> basis(degree_ + 1, 0.0);
        std::vector<double> left(degree_ + 1, 0.0);
        std::vector<double> right(degree_ + 1, 0.0);

        basis[0] = 1.0;

        for (int j = 1; j <= degree_; ++j)
        {
            left[j] = t - knots_[span + 1 - j];
            right[j] = knots_[span + j] - t;

            double saved = 0.0;

            for (int r = 0; r < j; ++r)
            {
                double basisMultiplier = basis[r] / (right[r + 1] + left[j - r]);
                basis[r] = saved + right[r + 1] * basisMultiplier;
                saved = left[j - r] * basisMultiplier;
            }

            basis[j] = saved;
        }

        return basis;
    }

public:
    std::vector<Point3D> controlPoints_;
    int n_; // control points - 1  
    int degree_; 
    std::vector<double> knots_;
};

void initCurveLength(const int numSamples,const BSplineCurve3D& curve,vector<double>&chord) 
{
    double len = 0.0;
    double dt = 1.0 / numSamples;
    chord.pb(0.0);

    for (int i = 0; i < numSamples; ++i)
    {
        double t0 = i * dt;
        double t1 = (i + 1) * dt;
        Point3D p0 = curve.evaluate(t0);
        Point3D p1 = curve.evaluate(t1);
        len += distance(p0, p1);
        chord.pb(len);
    }
    return;
}

inline double parameterAtLength(double length,const BSplineCurve3D&curve,vector<double>chord){
    const int numSamples = 1000;
    // double currentLength = 0.0;
    double dt = 1.0 / numSamples;
    double t = 0.0;
    if (length <= chord.front() + eps)
        return 0.0;
    else if (length >= chord.back() - eps)
        return 1.0;
    else
    {
        int low = 0;
        int high =chord.size()-1;
        int mid;

        while (high - low > 1)
        {
            mid = (high + low) >> 1;

            if (length < chord[mid] - eps)
                high = mid;
            else
                low = mid;
        }
        double t0 = dt * low;
        double t1 = dt * (low + 1);
        Point3D p0 = curve.evaluate(t0);
        Point3D p1 = curve.evaluate(t1);
        double remainingLength = length - chord[low];
        double segmentLength = distance(p0, p1);
        double alpha = remainingLength / segmentLength;
        t = t0 + alpha * dt;
        return t;
    }

}

bool render(int scale, const BSplineCurve3D& b,vector<Point3D>&params)
{

    int tempx = 0;
    int tempy = 0;
    int tempz = 0;
    int siz = params.size();
    for (size_t i = 0; i <siz; i++)
    {
        Point3D p = params[i];
        int x = round(p.x * scale);
        int y = round(p.y * scale);
        int z = round(p.z * scale);
        if (x < 0 || x>scale || y < 0 || y>scale || z < 0 || z>scale)
        {
            printf("error!\n");
            return false;
        }
        if (judge[x][y][z] && (x != tempx || y != tempy || z != tempz))
            return false;
        else
        {
            judge[x][y][z] = 1;
            tempx = x;
            tempy = y;
            tempz = z;
        }
    }
    return true;
}
inline std::vector<Point3D> sampleUniformly(int numSamples,const BSplineCurve3D&curve)
{
    // const auto& controlPoints_=curve.controlPoints_;
    std::vector<Point3D> samples;

    for (int i = 0; i < numSamples; ++i)
    {
        double t = i/(double)(numSamples-1);
        Point3D samplePoint = curve.evaluate(t);
        samples.pb(samplePoint);
       /* printf("%f %d %f   %f %f %f\n",i*segmentLength, i, t, samplePoint.x, samplePoint.y, samplePoint.z);*/
    }

    return samples;
}

inline std::vector<double> sampleEqualChordLengthParam(int revolution,int numSamples,const BSplineCurve3D&curve,vector<double>& chord){
    const auto& controlPoints_=curve.controlPoints_;
    std::vector<double> samples;
    double totalLength = chord.back();
    // printf("%f\n", totalLength);
    
    numSamples-=1;
    double segmentLength = totalLength / numSamples;
    double targetLength = 0.0;

    samples.pb(0);
    int pt = 0;
    double dt = 1.0 / revolution;
    for (int i = 1; i < numSamples; ++i)
    {
        targetLength = i * segmentLength;
        while (chord[pt] < targetLength - eps && pt < revolution)
            pt++;
        double t0 = (pt - 1) * dt;
        double t1 = pt * dt;
        Point3D p0 = curve.evaluate(t0);
        Point3D p1 = curve.evaluate(t1);
        double remainingLength = targetLength - chord[pt - 1];
        double seg = distance(p0, p1);
        double alpha = remainingLength / seg;
        double t = t0 + alpha * dt;
        samples.pb(t);
        /* printf("%f %d %f   %f %f %f\n",i*segmentLength, i, t, samplePoint.x, samplePoint.y, samplePoint.z);*/
    }
    samples.pb(1);

    return samples;
}

inline std::vector<double> sampleEqualChordLengthParamNoise(int revolution,int numSamples,double noise,const BSplineCurve3D&curve,vector<double>& chord){
    const auto& controlPoints_=curve.controlPoints_;
    std::vector<double> samples;
    double totalLength = chord.back();
    // printf("%f\n", totalLength);
    
    vector<double> initial_params(numSamples,0);
    std::default_random_engine generator;
    std::normal_distribution<double> normal(0,noise);
    const double clip_eps=1e-7;
    initial_params.back()=1;
    
    for (int i=1;i<numSamples-1;i++){
        initial_params[i]=normal(generator)+(double)i/(numSamples-1);
        if(initial_params[i]<clip_eps){
            initial_params[i]=clip_eps;
        }
        if(initial_params[i]>(1-clip_eps)){
            initial_params[i]=1-clip_eps;
        }
        // std::cout<<initial_params[i]<<" ";
    }
    // std::cout<<std::endl;
    std::sort(initial_params.begin(),initial_params.end());
    
    numSamples-=1;
    double targetLength = 0.0;

    samples.pb(0);
    int pt = 0;
    double dt = 1.0 / revolution;
    for (int i = 1; i < numSamples; ++i)
    {
        targetLength = totalLength*initial_params[i];
        while (chord[pt] < targetLength - eps && pt < revolution)
            pt++;
        double t0 = (pt - 1) * dt;
        double t1 = pt * dt;
        Point3D p0 = curve.evaluate(t0);
        Point3D p1 = curve.evaluate(t1);
        double remainingLength = targetLength - chord[pt - 1];
        double seg = distance(p0, p1);
        double alpha = remainingLength / seg;
        double t = t0 + alpha * dt;
        samples.pb(t);
        // std::cout<<"info:"<<" pt:"<<pt<<" chord[pt-1]:"<<chord[pt-1]<<" targetLen:"<<targetLength<<std::endl;
        // std::cout<<"info2:"<<" t:"<<t<<" t0:"<<t0<<" alpha:"<<alpha<<std::endl;
        // if(pt<revolution){
        //     std::cout<<" chord[pt]:"<<chord[pt]<<std::endl;
        // }
        /* printf("%f %d %f   %f %f %f\n",i*segmentLength, i, t, samplePoint.x, samplePoint.y, samplePoint.z);*/
    }
    samples.pb(1);

    return samples;
}

inline std::vector<Point3D> sampleEqualChordLength(int numSamples,const BSplineCurve3D&curve,vector<double>& chord)
{
    const auto& controlPoints_=curve.controlPoints_;
    std::vector<Point3D> samples;
    double totalLength = chord.back();
    // printf("%f\n", totalLength);
    double segmentLength = totalLength / numSamples ;
    double targetLength = 0.0;

    samples.pb(controlPoints_.front());
    for (int i = 1; i < numSamples; ++i)
    {
        targetLength = i * segmentLength;
        double t = parameterAtLength(targetLength,curve,chord);
        Point3D samplePoint = curve.evaluate(t);
        samples.pb(samplePoint);
        
        // std::cout<<"sample: "<<samplePoint.x<<","<<samplePoint.y<<","<< samplePoint.z<<std::endl; 
       /* printf("%f %d %f   %f %f %f\n",i*segmentLength, i, t, samplePoint.x, samplePoint.y, samplePoint.z);*/
    }
    samples.pb(controlPoints_.back());

    return samples;
}

std::vector<vector<double>> sampleEqualChordLength2(int degree,int numSamples,const vector<vector<double>> &ctrl_pts, const vector<double> &knots){
    Point3D tp;
    vector<Point3D> tcontrolp;
    int n = (int)ctrl_pts.size();
    
    // std::cout<<"degree: "<<degree<<std::endl;
    // std::cout<<"n: "<<n<<std::endl;
    for (int j = 0; j < n; j++)
    {
        /*printf("%f %f %f\n", c1[j][0], c1[j][1], c1[j][2]);*/
        tp.x = ctrl_pts[j][0];
        tp.y = ctrl_pts[j][1];
        tp.z = ctrl_pts[j][2];
        // std::cout<<"c:("<<tp.x<<", "<<tp.y<<", "<<tp.z<<")"<<std::endl;
        tcontrolp.pb(tp);
    }
    BSplineCurve3D tempB(degree,tcontrolp,knots);
    
    
    vector<double> chord;
    
    initCurveLength(numSamples, tempB, chord);
    auto tempP=sampleEqualChordLength(numSamples, tempB, chord);
    
    vector<vector<double>> ret;
    
    for(size_t i=0;i<tempP.size();i++){
        auto p=tempP[i];
        ret.push_back({p.x,p.y,p.z});
    }
    return ret;
}

std::vector<double> sampleEqualChordLengthParam_3D(int degree,int revolution,int numSamples,const vector<vector<double>> &ctrl_pts, const vector<double> &knots){
    Point3D tp;
    vector<Point3D> tcontrolp;
    int n = (int)ctrl_pts.size();
    
    // std::cout<<"degree: "<<degree<<std::endl;
    // std::cout<<"n: "<<n<<std::endl;
    for (int j = 0; j < n; j++)
    {
        /*printf("%f %f %f\n", c1[j][0], c1[j][1], c1[j][2]);*/
        tp.x = ctrl_pts[j][0];
        tp.y = ctrl_pts[j][1];
        tp.z = ctrl_pts[j][2];
        // std::cout<<"c:("<<tp.x<<", "<<tp.y<<", "<<tp.z<<")"<<std::endl;
        tcontrolp.pb(tp);
    }
    BSplineCurve3D tempB(degree,tcontrolp,knots);
    
    
    vector<double> chord;
    
    initCurveLength(revolution, tempB, chord);
    auto tempP=sampleEqualChordLengthParam(revolution,numSamples, tempB, chord);
    
    return  tempP;
}

std::vector<double> sampleEqualChordLengthParamNoised_3D(int degree,int revolution,int numSamples,double noise,const vector<vector<double>> &ctrl_pts, const vector<double> &knots){
    Point3D tp;
    vector<Point3D> tcontrolp;
    int n = (int)ctrl_pts.size();
    
    // std::cout<<"degree: "<<degree<<std::endl;
    // std::cout<<"n: "<<n<<std::endl;
    for (int j = 0; j < n; j++)
    {
        /*printf("%f %f %f\n", c1[j][0], c1[j][1], c1[j][2]);*/
        tp.x = ctrl_pts[j][0];
        tp.y = ctrl_pts[j][1];
        tp.z = ctrl_pts[j][2];
        // std::cout<<"c:("<<tp.x<<", "<<tp.y<<", "<<tp.z<<")"<<std::endl;
        tcontrolp.pb(tp);
    }
    BSplineCurve3D tempB(degree,tcontrolp,knots);
    
    
    vector<double> chord;
    
    initCurveLength(revolution, tempB, chord);
    // auto tempP=sampleEqualChordLengthParam(revolution,numSamples, tempB, chord);
    auto tempP=sampleEqualChordLengthParamNoise(revolution,numSamples, noise,tempB, chord);
    
    return  tempP;
}

bool isIntersect(int degree,int scale,int numSamples,bool chordEqual, const vector<vector<double>> &ctrl_pts, const vector<double> &knots){
    Point3D tp;
    vector<Point3D> tcontrolp;
    int n = (int)ctrl_pts.size();
    
    // std::cout<<"degree: "<<degree<<std::endl;
    // std::cout<<"n: "<<n<<std::endl;
    // std::cout<<scale<<std::endl;
    // std::cout<<chordEqual<<std::endl;

    for (int j = 0; j < n; j++)
    {
        /*printf("%f %f %f\n", c1[j][0], c1[j][1], c1[j][2]);*/
        tp.x = ctrl_pts[j][0];
        tp.y = ctrl_pts[j][1];
        tp.z = ctrl_pts[j][2];
        // std::cout<<"c:("<<tp.x<<", "<<tp.y<<", "<<tp.z<<")"<<std::endl;
        tcontrolp.pb(tp);
    }
    BSplineCurve3D tempB(degree,tcontrolp,knots);
    
    
    vector<Point3D> tempP;
    if(chordEqual){
        vector<double> chord;
        
        // std::cout<<"chord equal"<<std::endl;
        initCurveLength(numSamples, tempB, chord);
        tempP=sampleEqualChordLength(numSamples, tempB, chord);
    }
    else{
        // std::cout<<"uniform"<<std::endl;
        tempP=sampleUniformly(numSamples, tempB);
    }
    
    return render(scale, tempB, tempP);
    //for (int j = 0; j < c2.size(); j++)

    // return render(1000, tempB);
}