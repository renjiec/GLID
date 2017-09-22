#define  _CRT_SECURE_NO_WARNINGS

#include <mex.h>

#include <cuComplex.h>
#include <gpu/mxGPUArray.h>
#include "cuHarmonicDeform.cuh"


template<typename R>
R getFieldValueWithDefault(const mxArray* mat, const char* name, R defaultvalue)
{
    const mxArray *f = mat?mxGetField(mat, 0, name):nullptr;
    return f ? R(mxGetScalar(f)) : defaultvalue;
}

void mexError(const std::string& error)
{
    mexErrMsgTxt(("invalid input to mex: " + error).c_str());
}

const char* matClassNames[] = { "unknown", "cell", "struct", "logical", "char", "void", "double", "single", "int8", "uint8",
"int16", "uint16", "int32", "uint32", "int64", "uint64", "function", "opaque", "object" /* keep the last real item in the list */
};

const mxGPUArray* getFieldGPUArray(const mxArray* mat, const char* name, mxClassID verify_mxClassID, mxComplexity verify_complexity)
{
    const mxArray *m = mat ? mxGetField(mat, 0, name) : nullptr;

    if (!m) return nullptr;

    // has the field but not GPUArray
    if ( !mxIsGPUArray(m) )  mexError(std::string("cannot extract ") + name + ", not GPUArray?");

    if ( mxIsCell(m) )  mexError(name + std::string(" must be an array instead of cell!"));

    const mxGPUArray *data = mxGPUCreateFromMxArray(m);
    if (mxGPUGetClassID(data)    != verify_mxClassID) mexError(std::string("input array: ") + name + " should be of the same precision type as D2: " + matClassNames[verify_mxClassID]);
    if (mxGPUGetComplexity(data) != verify_complexity) mexError(std::string("input array: ") + name + " should be " + (verify_complexity==mxCOMPLEX?"complex":"real"));

    return data;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    // interface: cuHarmonic( D2, C2, bP2P, lambda, phipsyIters, cageOffsets, params );
    // params: optional, includes, isoetype, isoepow, nIter, enEvalsPerKernel, optimizationMethod

    mxInitGPU();   // Initialize the MathWorks GPU API.

    if (nrhs < 5)  mexError("not enough input");
    if (nrhs > 6 && !mxIsStruct(prhs[6])) mexError("params must be a struct");

    /* Throw an error if the input is not a GPU array. */
    if (!mxIsGPUArray(prhs[0]) || !mxIsGPUArray(prhs[1]) || !mxIsGPUArray(prhs[2]) || !mxIsGPUArray(prhs[4]))
        mexError("input must be GPU arrays");

    mxGPUArray const *bP2P = mxGPUCreateFromMxArray(prhs[2]);   // P2P target

    if (mxGPUGetNumberOfElements(bP2P) == 0) {
        mexWarnMsgTxt("No P2P is defined, solution is any global translation.");
        mxGPUDestroyGPUArray(bP2P);

        plhs[0] = mxDuplicateArray(prhs[4]);
        if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(0);

        return;
    }

    mxGPUArray const *D2 = mxGPUCreateFromMxArray(prhs[0]);     // derivative of Cauchy coordinates at iso energy samples
    mxGPUArray const *C2 = mxGPUCreateFromMxArray(prhs[1]);     // Cauchy coordinates for P2P constraints
    const double lambda = mxGetScalar(prhs[3]);
    mxGPUArray *const phipsyIters = mxGPUCopyFromMxArray(prhs[4]);

    std::vector<int> cageOffsets;
    if (nrhs > 5) {
        if (mxIsSparse(prhs[5]) || mxIsGPUArray(prhs[5]) || mxGetClassID(prhs[5]) != mxINT32_CLASS)
            mexError("input: cageoffset must be cpu array of int32 type");

        std::copy_n((int*)mxGetData(prhs[5]), mxGetNumberOfElements(prhs[5]), std::back_inserter(cageOffsets));
    }

    // all parameters
    const mxArray *params = nrhs>6?prhs[6]:nullptr;
    const int isoetype = getFieldValueWithDefault<int>(params, "isometric_energy_type", ISO_ENERGY_SYMMETRIC_DIRICHLET);
    const double isoepow = getFieldValueWithDefault<double>(params, "isometric_energy_power", 1);
    const int nIter = getFieldValueWithDefault<int>(params, "nIter", 1);
    const int enEvalsPerKernel = getFieldValueWithDefault<int>(params, "LS_energy_eval_per_kernel", 1);
    const int optimizationMethod = getFieldValueWithDefault<int>(params, "solver", OM_NEWTON_SPDH);
    const int reportIterStats = getFieldValueWithDefault<int>(params, "reportIterationStats", 1);
    const int linearSolvePref = getFieldValueWithDefault<int>(params, "linearSolvePref", LINEAR_SOLVE_PREFER_CHOLESKY);
    const double deltaFixSPDH = getFieldValueWithDefault<double>(params, "deltaFixSPDH", 1e-15);

    // Verify that invM really is a double array before extracting the pointer.
    if (mxGPUGetComplexity(D2) != mxCOMPLEX || mxGPUGetComplexity(C2) != mxCOMPLEX 
     || mxGPUGetComplexity(bP2P) != mxCOMPLEX || mxGPUGetComplexity(phipsyIters) != mxCOMPLEX)
        mexError("input array must be in complex numbers");

    const mwSize *dims = mxGPUGetDimensions(D2); // mxGPUGetNumberOfDimensions(D2) == 2;
    const int m = int(dims[0]);   // # samples
    const int n = int(dims[1]);   // dim Cauchy coordinates
    dims = mxGPUGetDimensions(C2);
    const int nP2P = int(dims[0]);

    const mxClassID runPrecision = mxGPUGetClassID(D2);


    const mxGPUArray *hessian_samples = getFieldGPUArray(params, "hessian_samples", mxINT32_CLASS, mxREAL);
    const int mh = hessian_samples? mxGPUGetNumberOfElements(hessian_samples):m;
    assert(mh <= m);

    //////////////////////////////////////////////////////////////////////////
    const mxGPUArray *sample_spacings =  getFieldGPUArray(params, "sample_spacings_half", runPrecision, mxREAL);
    const mxGPUArray* validationData_v = getFieldGPUArray(params, "v", runPrecision, mxCOMPLEX);
    const mxGPUArray* validationData_E2 =getFieldGPUArray(params, "E2", runPrecision, mxCOMPLEX);
    const mxGPUArray* validationData_L = getFieldGPUArray(params, "L", runPrecision, mxREAL);
    const mxGPUArray* validationData_nextSample = getFieldGPUArray(params, "nextSampleInSameCage", mxINT32_CLASS, mxREAL);

    // Now that we have verified the data type, extract a pointer to the input data on the device.

    const mxClassID inputPrecisonTypes[] = { mxGPUGetClassID(C2), mxGPUGetClassID(bP2P), mxGPUGetClassID(phipsyIters) };
    for (auto pt:inputPrecisonTypes) if (pt != runPrecision) mexError("input arrays should be all of the same precision type");

    const int *hessian_samples_rawp = hessian_samples ? (const int *)(mxGPUGetDataReadOnly(hessian_samples)) : nullptr;
    const void *sample_spacings_rawp = sample_spacings ? mxGPUGetDataReadOnly(sample_spacings) : 0;

    std::vector<double> allStats;
    if (runPrecision == mxDOUBLE_CLASS){
        using Complex = cuDoubleComplex;
        using real = double;

        HarmonicMapValidationInput<Complex, real> validation_data;
        
        if (validationData_v && validationData_E2 && validationData_L) {
            validation_data = { (const Complex*)mxGPUGetDataReadOnly(validationData_v),
                                (const Complex*)mxGPUGetDataReadOnly(validationData_E2),
                                (const real*)mxGPUGetDataReadOnly(validationData_L),
                                (const int*)mxGPUGetDataReadOnly(validationData_nextSample) };
        }

        //////////////////////////////////////////////////////////////////////////
        allStats = cuHarmonicDeform<Complex, real>((Complex const *)(mxGPUGetDataReadOnly(D2)),
            (Complex const *)(mxGPUGetDataReadOnly(C2)),
            (Complex const *)(mxGPUGetDataReadOnly(bP2P)),
            (Complex *)(mxGPUGetData(phipsyIters)),
            hessian_samples_rawp,
            m, mh, n, cageOffsets, nP2P, isoetype, isoepow, lambda, nIter, 
            validation_data,
            (const real*)sample_spacings_rawp, 
            enEvalsPerKernel, optimizationMethod,
            reportIterStats, linearSolvePref, deltaFixSPDH);

    }
    else if (runPrecision == mxSINGLE_CLASS) {
        using Complex = cuFloatComplex;
        using real = float;

        HarmonicMapValidationInput<Complex, real> validation_data;
        
        if (validationData_v && validationData_E2 && validationData_L) {
            validation_data = { (const Complex*)mxGPUGetDataReadOnly(validationData_v),
                                (const Complex*)mxGPUGetDataReadOnly(validationData_E2),
                                (const real*)mxGPUGetDataReadOnly(validationData_L),
                                (const int*)mxGPUGetDataReadOnly(validationData_nextSample) };
        }

        //////////////////////////////////////////////////////////////////////////
        allStats = cuHarmonicDeform<Complex, real>((Complex const *)(mxGPUGetDataReadOnly(D2)),
            (Complex const *)(mxGPUGetDataReadOnly(C2)),
            (Complex const *)(mxGPUGetDataReadOnly(bP2P)),
            (Complex *)(mxGPUGetData(phipsyIters)),
            hessian_samples_rawp,
            m, mh, n, cageOffsets, nP2P, isoetype, isoepow, lambda, nIter, 
            validation_data,
            (const real*)sample_spacings_rawp, 
            enEvalsPerKernel, optimizationMethod,
            reportIterStats, linearSolvePref, deltaFixSPDH);
    }

    // Wrap the result up as a MATLAB gpuArray for return.
    plhs[0] = mxGPUCreateMxArrayOnGPU(phipsyIters);
    if (nlhs > 1) {
        if (allStats.size() == 1)
            plhs[1] = mxCreateDoubleScalar(allStats[0]);
        else {
            plhs[1] = mxCreateDoubleMatrix(allStats.size() / (nIter+1), (nIter+1), mxREAL);
            std::copy_n(allStats.cbegin(), mxGetNumberOfElements(plhs[1]), mxGetPr(plhs[1]));
        }
    }

    // * The mxGPUArray pointers are host-side structures that refer to device
    // * data. These must be destroyed before leaving the MEX function.
    mxGPUDestroyGPUArray(D2);
    mxGPUDestroyGPUArray(C2);
    mxGPUDestroyGPUArray(bP2P);
    mxGPUDestroyGPUArray(phipsyIters);

    if (hessian_samples) mxGPUDestroyGPUArray(hessian_samples);
    if (sample_spacings) mxGPUDestroyGPUArray(sample_spacings);

    if (validationData_v)  mxGPUDestroyGPUArray(validationData_v);
    if (validationData_E2) mxGPUDestroyGPUArray(validationData_E2);
    if (validationData_L)  mxGPUDestroyGPUArray(validationData_L);
    if (validationData_nextSample) mxGPUDestroyGPUArray(validationData_nextSample);
}
