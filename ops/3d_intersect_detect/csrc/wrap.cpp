#include <cstddef>
#include <iostream>
#include <iterator>
#include <ostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>
#include <vector>
#include "detect.hpp"

namespace py = pybind11;

using namespace pybind11::literals;

py::array sampleEqualChordLengthNumpy(int degree,int numSamples,py::array_t<double> control_pts,py::array_t<double> knots){
    // std::cout<< "do request"<<std::endl;
    py::buffer_info buf = control_pts.request();

    // Check if the matrix is 2D
    // std::cout<<"check dim is 2"<<std::endl;
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix");
    }

    // Get the shape and size of the matrix
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    size_t size = buf.size;
    
    // std::cout<<"rows: "<<rows<<std::endl;
    // std::cout<<"cols: "<<cols<<std::endl;
    // std::cout<<"size: "<<size<<std::endl;

    // Get the pointer to the data
    // std::cout<<"Get the pointer to the data"<<std::endl;
    double* data = static_cast<double*>(buf.ptr);

    // Create a nested std::vector from the data
    // std::cout<<"Create nested vector from data"<<std::endl;
    std::vector<std::vector<double>> ctrl(rows, std::vector<double>(cols));
    // std::cout<<"Copying..."<<std::endl;
    for (size_t i = 0; i < rows; i++){
        std::copy(data + i * cols, data + i * cols + cols, ctrl[i].begin());}
    // std::cout<<"Copy done"<<std::endl; 
    // for (auto i=0;i<size;i++){
    //     std::cout<<data[i]<<" ";
    // }
    // std::cout<<std::endl;


    // std::cout<<"Create Knots"<<std::endl;
    buf = knots.request();
    // std::cout<<"knot ndim: "<<buf.ndim<<std::endl;
    
    size=buf.shape[0];
    data = static_cast<double*>(buf.ptr);
    rows=buf.shape[0];
    std::vector<double> k(rows);
    std::copy(data, data + size, k.begin());
    
    // std::cout<<"do intesect detection"<<std::endl;
    auto Ps=sampleEqualChordLength2(degree, numSamples, ctrl, k);
    
    // for (size_t i =0;i<Ps.size();i++){
    //     auto p=Ps[i];
    //     std::cout<<"p: "<<std::endl;
    //     for (size_t j=0;j<p.size();j++){
    //         std::cout<<p[j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    vector<double> Ps_plain;
    for (size_t i =0;i<Ps.size();i++){
        auto p=Ps[i];
        for (size_t j=0;j<p.size();j++){
            Ps_plain.push_back(p[j]);
        }
    }
    
    // std::cout<<"create np array"<<std::endl;
    py::array_t<double> npArray(Ps_plain.size(),Ps_plain.data());

    // std::cout<<"reshape"<<std::endl;
    return npArray.reshape({Ps.size(),Ps[0].size()});

}

vector<vector<double>> get2DVectorFromNumpy(py::array_t<double>& control_pts){
    // std::cout<< "do request"<<std::endl;
    py::buffer_info buf = control_pts.request();

    // Check if the matrix is 2D
    // std::cout<<"check dim is 2"<<std::endl;
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix");
    }

    // Get the shape and size of the matrix
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    size_t size = buf.size;
    
    // std::cout<<"rows: "<<rows<<std::endl;
    // std::cout<<"cols: "<<cols<<std::endl;
    // std::cout<<"size: "<<size<<std::endl;

    // Get the pointer to the data
    // std::cout<<"Get the pointer to the data"<<std::endl;
    double* data = static_cast<double*>(buf.ptr);

    // Create a nested std::vector from the data
    // std::cout<<"Create nested vector from data"<<std::endl;
    std::vector<std::vector<double>> ctrl(rows, std::vector<double>(cols));
    // std::cout<<"Copying..."<<std::endl;
    for (size_t i = 0; i < rows; i++){
        std::copy(data + i * cols, data + i * cols + cols, ctrl[i].begin());}
    // std::cout<<"Copy done"<<std::endl; 
    // for (auto i=0;i<size;i++){
    //     std::cout<<data[i]<<" ";
    // }
    // std::cout<<std::endl;
    return ctrl;
}

vector<double> get1DVectorFromNumpy(py::array_t<double> array){
    // std::cout<< "do request"<<std::endl;
    py::buffer_info buf = array.request();

    // Check if the matrix is 2D
    // std::cout<<"check dim is 2"<<std::endl;
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be a 1D matrix");
    }

    // Get the shape and size of the matrix
    size_t size = buf.shape[0];
    
    // std::cout<<"rows: "<<rows<<std::endl;
    // std::cout<<"cols: "<<cols<<std::endl;
    // std::cout<<"size: "<<size<<std::endl;

    // Get the pointer to the data
    // std::cout<<"Get the pointer to the data"<<std::endl;
    double* data = static_cast<double*>(buf.ptr);

    // Create a nested std::vector from the data
    // std::cout<<"Create nested vector from data"<<std::endl;
    std::vector<double> ctrl(size);
    // std::cout<<"Copying..."<<std::endl;
    std::copy(data, data + size, ctrl.begin());
    // std::cout<<"Copy done"<<std::endl; 
    // for (auto i=0;i<size;i++){
    //     std::cout<<data[i]<<" ";
    // }
    // std::cout<<std::endl;
    return ctrl;
}
py::array sampleEqualChordLengthParam2DNumpy(int degree,int revolution,int numSamples,py::array_t<double> control_pts,py::array_t<double> knots){
    // std::cout<< "do request"<<std::endl;
    py::buffer_info buf = control_pts.request();

    // Check if the matrix is 2D
    // std::cout<<"check dim is 2"<<std::endl;
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix");
    }

    // Get the shape and size of the matrix
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    size_t size = buf.size;
    
    // std::cout<<"rows: "<<rows<<std::endl;
    // std::cout<<"cols: "<<cols<<std::endl;
    // std::cout<<"size: "<<size<<std::endl;

    // Get the pointer to the data
    // std::cout<<"Get the pointer to the data"<<std::endl;
    double* data = static_cast<double*>(buf.ptr);

    // Create a nested std::vector from the data
    // std::cout<<"Create nested vector from data"<<std::endl;
    std::vector<std::vector<double>> ctrl(rows, std::vector<double>(cols));
    // std::cout<<"Copying..."<<std::endl;
    for (size_t i = 0; i < rows; i++){
        std::copy(data + i * cols, data + i * cols + cols, ctrl[i].begin());}
    // std::cout<<"Copy done"<<std::endl; 
    // for (auto i=0;i<size;i++){
    //     std::cout<<data[i]<<" ";
    // }
    // std::cout<<std::endl;


    // std::cout<<"Create Knots"<<std::endl;
    buf = knots.request();
    // std::cout<<"knot ndim: "<<buf.ndim<<std::endl;
    
    size=buf.shape[0];
    data = static_cast<double*>(buf.ptr);
    rows=buf.shape[0];
    std::vector<double> k(rows);
    std::copy(data, data + size, k.begin());
    
    // std::cout<<"do intesect detection"<<std::endl;
    auto Ps=sampleEqualChordLengthParam_2D(degree,revolution, numSamples, ctrl, k);
    
    // for (size_t i =0;i<Ps.size();i++){
    //     auto p=Ps[i];
    //     std::cout<<"p: "<<std::endl;
    //     for (size_t j=0;j<p.size();j++){
    //         std::cout<<p[j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    
    // std::cout<<"create np array"<<std::endl;
    py::array_t<double> npArray(Ps.size(),Ps.data());

    // std::cout<<"reshape"<<std::endl;
    return npArray;
}

py::array sampleEqualChordLengthParam3DNumpy(int degree,int revolution,int numSamples,py::array_t<double> control_pts,py::array_t<double> knots){
    // std::cout<< "do request"<<std::endl;
    py::buffer_info buf = control_pts.request();

    // Check if the matrix is 2D
    // std::cout<<"check dim is 2"<<std::endl;
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix");
    }

    // Get the shape and size of the matrix
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    size_t size = buf.size;
    
    // std::cout<<"rows: "<<rows<<std::endl;
    // std::cout<<"cols: "<<cols<<std::endl;
    // std::cout<<"size: "<<size<<std::endl;

    // Get the pointer to the data
    // std::cout<<"Get the pointer to the data"<<std::endl;
    double* data = static_cast<double*>(buf.ptr);

    // Create a nested std::vector from the data
    // std::cout<<"Create nested vector from data"<<std::endl;
    std::vector<std::vector<double>> ctrl(rows, std::vector<double>(cols));
    // std::cout<<"Copying..."<<std::endl;
    for (size_t i = 0; i < rows; i++){
        std::copy(data + i * cols, data + i * cols + cols, ctrl[i].begin());}
    // std::cout<<"Copy done"<<std::endl; 
    // for (auto i=0;i<size;i++){
    //     std::cout<<data[i]<<" ";
    // }
    // std::cout<<std::endl;


    // std::cout<<"Create Knots"<<std::endl;
    buf = knots.request();
    // std::cout<<"knot ndim: "<<buf.ndim<<std::endl;
    
    size=buf.shape[0];
    data = static_cast<double*>(buf.ptr);
    rows=buf.shape[0];

    std::vector<double> k(rows);
    std::copy(data, data + size, k.begin());
    
    // std::cout<<"do intesect detection"<<std::endl;
    auto Ps=sampleEqualChordLengthParam_3D(degree,revolution, numSamples, ctrl, k);
    
    // for (size_t i =0;i<Ps.size();i++){
    //     auto p=Ps[i];
    //     std::cout<<"p: "<<std::endl;
    //     for (size_t j=0;j<p.size();j++){
    //         std::cout<<p[j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    
    // std::cout<<"create np array"<<std::endl;
    py::array_t<double> npArray(Ps.size(),Ps.data());

    // std::cout<<"reshape"<<std::endl;
    return npArray;
}

py::array sampleEqualChordLengthParam2DNumpyNoised(int degree,int revolution,int numSamples,double noise,py::array_t<double> control_pts,py::array_t<double> knots){
    auto ctrl=get2DVectorFromNumpy(control_pts);
    auto k=get1DVectorFromNumpy(knots);
    auto Ps=sampleEqualChordLengthParamNoised_2D(degree,revolution, numSamples, noise,ctrl, k);
    py::array_t<double> npArray(Ps.size(),Ps.data());

    return npArray;
}

py::array sampleEqualChordLengthParam3DNumpyNoised(int degree,int revolution,int numSamples,double noise,py::array_t<double> control_pts,py::array_t<double> knots){
    auto ctrl=get2DVectorFromNumpy(control_pts);
    auto k=get1DVectorFromNumpy(knots);
    auto Ps=sampleEqualChordLengthParamNoised_3D(degree,revolution, numSamples, noise,ctrl, k);
    py::array_t<double> npArray(Ps.size(),Ps.data());

    return npArray;
}

bool isIntersectNumpy(int degree,int scale,int num_samples,bool isEqualChord,py::array_t<double> control_pts,py::array_t<double> knots){
    // std::cout<< "do request"<<std::endl;
    py::buffer_info buf = control_pts.request();
    
    // std::cout<<"is EqualChord "<<isEqualChord<<std::endl;

    // Check if the matrix is 2D
    // std::cout<<"check dim is 2"<<std::endl;
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix");
    }

    // Get the shape and size of the matrix
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    size_t size = buf.size;
    
    // std::cout<<"rows: "<<rows<<std::endl;
    // std::cout<<"cols: "<<cols<<std::endl;
    // std::cout<<"size: "<<size<<std::endl;

    // Get the pointer to the data
    // std::cout<<"Get the pointer to the data"<<std::endl;
    double* data = static_cast<double*>(buf.ptr);

    // Create a nested std::vector from the data
    // std::cout<<"Create nested vector from data"<<std::endl;
    std::vector<std::vector<double>> ctrl(rows, std::vector<double>(cols));
    // std::cout<<"Copying..."<<std::endl;
    for (size_t i = 0; i < rows; i++){
        std::copy(data + i * cols, data + i * cols + cols, ctrl[i].begin());}
    // std::cout<<"Copy done"<<std::endl; 
    // for (auto i=0;i<size;i++){
    //     std::cout<<data[i]<<" ";
    // }
    // std::cout<<std::endl;


    // std::cout<<"Create Knots"<<std::endl;
    buf = knots.request();
    // std::cout<<"knot ndim: "<<buf.ndim<<std::endl;
    
    size=buf.shape[0];
    data = static_cast<double*>(buf.ptr);
    rows=buf.shape[0];
    std::vector<double> k(rows);
    std::copy(data, data + size, k.begin());
    
    // std::cout<<"do intesect detection"<<std::endl;
    return isIntersect(degree,scale,num_samples,isEqualChord,ctrl,k);
}

bool isIntersect2DNumpy(int degree,int scale,int num_samples,bool isEqualChord,py::array_t<double> control_pts,py::array_t<double> knots){
    // std::cout<< "do request"<<std::endl;
    py::buffer_info buf = control_pts.request();
    
    // std::cout<<"is EqualChord "<<isEqualChord<<std::endl;

    // Check if the matrix is 2D
    // std::cout<<"check dim is 2"<<std::endl;
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix");
    }

    // Get the shape and size of the matrix
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    size_t size = buf.size;
    
    // std::cout<<"rows: "<<rows<<std::endl;
    // std::cout<<"cols: "<<cols<<std::endl;
    // std::cout<<"size: "<<size<<std::endl;

    // Get the pointer to the data
    // std::cout<<"Get the pointer to the data"<<std::endl;
    double* data = static_cast<double*>(buf.ptr);

    // Create a nested std::vector from the data
    // std::cout<<"Create nested vector from data"<<std::endl;
    std::vector<std::vector<double>> ctrl(rows, std::vector<double>(cols));
    // std::cout<<"Copying..."<<std::endl;
    for (size_t i = 0; i < rows; i++){
        std::copy(data + i * cols, data + i * cols + cols, ctrl[i].begin());}
    // std::cout<<"Copy done"<<std::endl; 
    // for (auto i=0;i<size;i++){
    //     std::cout<<data[i]<<" ";
    // }
    // std::cout<<std::endl;


    // std::cout<<"Create Knots"<<std::endl;
    buf = knots.request();
    // std::cout<<"knot ndim: "<<buf.ndim<<std::endl;
    
    size=buf.shape[0];
    data = static_cast<double*>(buf.ptr);
    rows=buf.shape[0];
    std::vector<double> k(rows);
    std::copy(data, data + size, k.begin());
    
    // std::cout<<"do intesect detection"<<std::endl;
    return isIntersect2D(degree,scale,num_samples,isEqualChord,ctrl,k);
}

PYBIND11_MODULE(opsIntersectDetect3D, m) {
    m.def("isIntersect", &isIntersectNumpy, R"pbdoc(
        To tell if a curve is intersected with a surface, we need to calculate the intersection points of the curves.
    )pbdoc");

    m.def("isIntersect2D", &isIntersect2DNumpy, R"pbdoc(
        To tell if a curve is intersected with a surface, we need to calculate the intersection points of the curves.
    )pbdoc");

    m.def("sampleEqualChordLength", &sampleEqualChordLengthNumpy, R"pbdoc(
        sample points with equal chord lengt
    )pbdoc");
    m.def("sampleEqualChordLengthParam2D", &sampleEqualChordLengthParam2DNumpy, R"pbdoc(
        sample points with equal chord lengt
    )pbdoc");
    m.def("sampleEqualChordLengthParam2DNoised", &sampleEqualChordLengthParam2DNumpyNoised, R"pbdoc(
        sample points with equal chord lengt + noise
    )pbdoc");
    m.def("sampleEqualChordLengthParam3D", &sampleEqualChordLengthParam3DNumpy, R"pbdoc(
        sample points with equal chord lengt
    )pbdoc");
    m.def("sampleEqualChordLengthParam3DNoised", &sampleEqualChordLengthParam3DNumpyNoised, R"pbdoc(
        sample points with equal chord lengt + noise
    )pbdoc");
}