#include <vector>
using std::vector;

bool isIntersect(int degree,int scale,int numSamples,bool chordEqual, const vector<vector<double>> &ctrl_pts, const vector<double> &knots);
bool isIntersect2D(int degree,int scale,int numSamples,bool chordEqual, const vector<vector<double>> &ctrl_pts, const vector<double> &knots);
std::vector<double> sampleEqualChordLengthParam_2D(int degree,int revolution,int numSamples,const vector<vector<double>> &ctrl_pts, const vector<double> &knots);
std::vector<vector<double>> sampleEqualChordLength2(int degree,int numSamples,const vector<vector<double>> &ctrl_pts, const vector<double> &knots);
std::vector<double> sampleEqualChordLengthParamNoised_2D(int degree,int revolution,int numSamples,double noise,const vector<vector<double>> &ctrl_pts, const vector<double> &knots);
std::vector<double> sampleEqualChordLengthParam_3D(int degree,int revolution,int numSamples,const vector<vector<double>> &ctrl_pts, const vector<double> &knots);
std::vector<double> sampleEqualChordLengthParamNoised_3D(int degree,int revolution,int numSamples,double noise,const vector<vector<double>> &ctrl_pts, const vector<double> &knots);