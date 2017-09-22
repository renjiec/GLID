#include <cstdarg>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <cassert>
#include <algorithm>

 
inline __host__ __device__ int ceildiv(int m, int n) { return (m - 1) / n + 1; }

#ifdef __CUDA_ARCH__ 
#undef printf // undef mexPrintf
#endif

inline void cudaCheckError(const char* file, int line)
{
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "CUDA error at %s:%i code=%d(%s)\n", file, line, int(err), cudaGetErrorString(err));
        cudaDeviceReset();
        throw std::exception("cuda runtime error");
    }
}

#if defined(_DEBUG) && !defined(__CUDACC__)
#define CUDA_CHECK_ERROR cudaCheckError(__FILE__, __LINE__);
#else
#define CUDA_CHECK_ERROR  
#endif

template< typename T >
void checkCuda(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();

        // Make sure we call CUDA Device Reset before exiting
        throw std::exception("cuda runtime error");
    }
}

#define checkCudaErrors(val)           checkCuda ( (val), #val, __FILE__, __LINE__ )

template<class R, int n=1>
__host__ __device__ void print_gpu_value(const R *d_v, const char* valname = "value", bool newline = true)
{
#if defined(_DEBUG) && !defined(__CUDA_ARCH__)      // do not call the following on cuda kernel
    R v[n];
    cudaMemcpy(v, d_v, sizeof(R)*n, cudaMemcpyDeviceToHost);
    printf("%20s = ", valname);
    for (int i = 0; i < n; i++) 
        printf("%6.5e%s", v[i], (i < n - 1) ? "\t" : (newline ? "\n" : ""));
#endif
}


template<class R>
__device__ void gpu_print(const R v, const char* valname = "value", bool newline = true)
{
#ifdef _DEBUG
    static_assert(!std::is_pointer<R>::value, "print pointer address?");
    printf("%20s = %10e%s", valname, float(v), newline ? "\n" : "\t");
#endif
}


#define ensure(cond, msg) if (!(cond)) fprintf(stderr, (msg+std::string(" at file %s, line %d\n")).c_str(), __FILE__, __LINE__);

#define EnableForComplexReturn(ret) __forceinline__ __host__ __device__ \
std::enable_if_t<std::is_same<Complex, cuFloatComplex>::value || std::is_same<Complex, cuDoubleComplex>::value, ret>

template<class Complex>
EnableForComplexReturn(Complex) conj(const Complex& a) { return Complex{ a.x, -a.y }; }
template<class Complex>
EnableForComplexReturn(Complex) operator+(const Complex& a, const Complex& b) { return Complex{ a.x+b.x, a.y+b.y }; }
template<class Complex>
EnableForComplexReturn(Complex) operator-(const Complex& a, const Complex& b) { return Complex{ a.x-b.x, a.y-b.y }; }
template<class Complex, class real>
EnableForComplexReturn(Complex) operator*(const Complex& a, const real b) { using R = decltype(a.x); return Complex{ R(a.x*b), R(a.y*b) }; }
template<class Complex>
EnableForComplexReturn(Complex) operator*(const Complex& a, const Complex& b) { return Complex{ a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x }; }
template<class Complex>
EnableForComplexReturn(Complex) operator/(const Complex& a, const Complex& b) { return a*conj(b)*(1/abs2(b)); }
template<class Complex>
EnableForComplexReturn(Complex)& operator+=(Complex& a, const Complex& b) { a.x += b.x; a.y += b.y; return a; }
template<class Complex>
EnableForComplexReturn(double) abs2(const Complex v) { return v.x*v.x + v.y*v.y; };
template<class Complex>
EnableForComplexReturn(double) abs(const Complex v) { return sqrt(v.x*v.x + v.y*v.y); };
template<class Complex>
EnableForComplexReturn(double) dot(const Complex &a, const Complex &b) { return a.x*b.x + a.y*b.y; };
template<class real>
__forceinline__ __host__ __device__ double sqr(real x) { return x*x; };
template<class real>
__forceinline__ __host__ __device__ double pow3(real x) { return x*x*x; };
template<class real>
__forceinline__ __host__ __device__ double pow4(real x) { return sqr( sqr(x) ); };


#undef EnableForComplexReturn    

template<class T>
inline cudaError_t myCopy_n(const T *src, int n, T *dst, cudaMemcpyKind cpydir=cudaMemcpyDeviceToDevice) { return cudaMemcpyAsync(dst, src, sizeof(T)*n, cpydir); }

template<class T>
inline T copyValFromGPU(const T *src) {
    T val;
    ensure( cudaSuccess ==  cudaMemcpyAsync(&val, src, sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy error"); 
    return val;
}

template<class T>
inline cudaError_t myZeroFill(T *dst, int n = 1) { return cudaMemsetAsync(dst, 0, sizeof(T)*n); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__forceinline__ __device__ void atomicMinPositive(float *t, float x)
{
    //atomicMin((unsigned int*)t,  __float_as_uint(max(0., x))); // todo: atomicMin for float?  the current code works only if all numbers are non-negative
    atomicMin((unsigned int*)t,  __float_as_uint(x));
}

__forceinline__ __device__ void atomicMaxPositive(float *t, float x)
{
    atomicMax((unsigned int*)t,  __float_as_uint(x));
}

//https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
// For all float & double atomics:
//      Must do the compare with integers, not floating point,
//      since NaN is never equal to any other NaN
//
// float atomicMin
__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

#if __CUDA_ARCH__ >= 130

__forceinline__ __device__ void atomicMinPositive(double *t, double x)
{
    //atomicMin((unsigned long long*)t, __double_as_longlong(max(0., x)));
    atomicMin((unsigned long long*)t, __double_as_longlong(x));
}

__forceinline__ __device__ void atomicMaxPositive(double *t, double x)
{
    atomicMax((unsigned long long*)t, __double_as_longlong(x));
}

// double atomicMin
__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}
#endif

template<class R>
__global__ void myscaling(R* x, R s) { *x *= s; }

template<class Complex>
__global__ void conjugate_inplace(Complex *f, int n) 
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= n) return;
	//*((scalar*)(f+i)+1) *= -1; // todo: optimize
    f[i] = conj(f[i]);
}

template<class R>
__global__ void clear_nans(const R* x, R *y, int n, R resetVal = 0)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)  return;

    y[i] = isfinite(x[i]) ? x[i] : resetVal;
}

template<class R>
__global__ void addToMatrixDiagonal(R *A, R delta, int n, int lda = -1)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= n) return;

    if (lda < 0) lda = n;

    A[i*lda + i] += delta;
}

template<class R>
__global__ void squareMatrixCopyWithSkip(const R *x, int n, int n2, int skip, R *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= n2 || j >= n2) return;
    y[i + j*n2] = x[(i<skip?i:i+1) + (j<skip?j:j+1)*n];
}

template<class Complex, class real>
__global__ void vectorComplex2Real(const Complex *x, int n, real *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Complex &z = x[i];
    y[i] = z.x;
    y[i + n] = z.y;
}

template<class Complex, class real>
__global__ void vectorReal2Complex(const real *x, int ldx, int n, Complex *y, real scale=1.)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    y[i] = Complex({ x[i]*scale, x[i + ldx]*scale });
}

template<class Complex, class real>
__global__ void matrixRowsComplex2Real(const Complex *x, int m, int n, real *y, const int *rows=nullptr, int k = -1)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (!rows || k <= 0) k = m;
    if (i >= k || j >= n) return;

    const Complex &z = x[ (rows?rows[i]:i) + j*m ];

    const int k2 = k * 2;  // for writing to a bigger matrix
    y[i + j*k2]     =  z.x;
    y[i + (j+n)*k2] = -z.y;
    y[i+k + j*k2]   =  z.y;
    y[i+k + (j+n)*k2]= z.x;
}


template<class R>
__global__ void swapMatrixInMemory(R *const x, R *const y, int m, int n, int lda)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;

    int k = i + j*lda;
    R z = x[k];
    x[k] = y[k];
    y[k] = z;
}


template<class T>
struct cuVector 
{
    T* pdata = nullptr;
    int len;

    cuVector(const cuVector&) = delete;
    cuVector& operator=(const cuVector&) = delete;
    cuVector(cuVector&& t):len(0), pdata(nullptr) { std::swap(pdata, t.pdata); std::swap(len, t.len); }

    cuVector(int n = 0, const T* srcbegin=nullptr) :len(n),pdata(nullptr) {
        if (len > 0) {
            checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(T)*len));
            if (srcbegin)
                checkCudaErrors(cudaMemcpyAsync(pdata, srcbegin, len * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    cuVector(const std::vector<T> &v) :len(v.size()),pdata(nullptr) {
        if (len > 0) {
            checkCudaErrors(cudaMalloc((void **)&pdata, sizeof(T)*len));
            checkCudaErrors(cudaMemcpyAsync(pdata, v.data(), len * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    ~cuVector() { if (pdata) checkCudaErrors(cudaFree(pdata)); }

    int size() const { return len; }
    bool empty() const { return len == 0; }

    void clear() {
        if (pdata) {
            checkCudaErrors(cudaFree(pdata));
            pdata = nullptr;
            len = 0;
        }
    }

    T* data() { return pdata; }
    const T* data() const { return pdata; }

    cuVector& operator =(const std::vector<T> &v){
        resize(v.size(), false);
        checkCudaErrors(cudaMemcpyAsync(pdata, v.data(), len*sizeof(T), cudaMemcpyHostToDevice));
        return *this;
    }

    cuVector& operator =(cuVector&& t) { std::swap(pdata, t.pdata); std::swap(len, t.len); return *this; }

    void zero_fill() {checkCudaErrors(cudaMemsetAsync(pdata, 0, sizeof(T)*len)); }

    void to_host(T* dst, int n) const {
        assert(n <= len); 
        checkCudaErrors(cudaMemcpyAsync(dst, pdata, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    operator std::vector<T>() const { std::vector<T> v(len); to_host(v.data(), len); return v; }

    void resize(int n, bool copyData=true) {
        if (n == len) return;

        cuVector<T> v2(n);
        if (copyData && len > 0) checkCudaErrors(cudaMemcpyAsync(v2.data(), pdata, std::min(len, n) * sizeof(T), cudaMemcpyDeviceToDevice));

        std::swap(*this, v2);
    }
};


template<class R>
struct cuSolverDN
{
    cusolverDnHandle_t handle;
    cuVector<R> buffer;
    cuVector<R> A;
    cuVector<int> info;
    int n = 0;

    cuSolverDN() :handle(nullptr), info(1) { info.zero_fill(); }
    virtual ~cuSolverDN() {  if(handle)   cusolverDnDestroy(handle); }

    void init(int neq) { n = neq; A.resize(n*n, false); }

    virtual int factor(const R* A0 = nullptr) = 0;
    virtual int solve(R *const) = 0;
};



#define ChooseCusolverDnFunc(fun, dfun, sfun) \
    using fun##type = std::conditional_t<runDoublePrecision, decltype(&cusolverDn##dfun), decltype(&cusolverDn##sfun)>; \
    const auto fun = runDoublePrecision ? (fun##type)&cusolverDn##dfun : (fun##type)&cusolverDn##sfun;

template<class R>
struct cuLUSolverDn : public cuSolverDN<R>
{
    cuVector<int> ipiv; // pivot for LU solver

    cuLUSolverDn()  {}
    ~cuLUSolverDn() {}

    int factor(const R* A0 = nullptr) {
        if(!handle)   cusolverDnCreate(&handle);

        const bool runDoublePrecision = std::is_same<R, double>::value;
        ChooseCusolverDnFunc(getrf_bufferSize, Dgetrf_bufferSize, Sgetrf_bufferSize);
        ChooseCusolverDnFunc(getrf, Dgetrf, Sgetrf);

        A.resize(n*n, false);   // allocate memory if necessary
        if (A0 && A0 != A.data())  myCopy_n(A0, n*n, A.data());

        int bufferSize = 0;
        getrf_bufferSize(handle, n, n, A.data(), n, &bufferSize);

        buffer.resize(bufferSize);
        ipiv.resize(n);

        //cusolverDnZgetrf
        getrf(handle, n, n, A.data(), n, buffer.data(), ipiv.data(), info.data());
        CUDA_CHECK_ERROR;

        return 0;
    }

    int solve(R *const b) {

        const bool runDoublePrecision = std::is_same<R, double>::value;
        ChooseCusolverDnFunc(getrs, Dgetrs, Sgetrs);

        // b will be rewritten in place on return
        getrs(handle, CUBLAS_OP_N, n, 1, A.data(), n, ipiv.data(), b, n, info.data());

        return 0;
    }
};


template<class R>
struct cuCholSolverDn: public cuSolverDN<R>
{
    static const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    cuCholSolverDn() {}
    virtual ~cuCholSolverDn() {}

    int factor(const R* A0 = nullptr) {
        if(!handle)   cusolverDnCreate(&handle);

        const bool runDoublePrecision = std::is_same<R, double>::value;

        ChooseCusolverDnFunc(potrf_bufferSize, Dpotrf_bufferSize, Spotrf_bufferSize);
        ChooseCusolverDnFunc(potrf, Dpotrf, Spotrf);

        A.resize(n*n, false);   // allocate memory if necessary
        if (A0 && A0 != A.data())  myCopy_n(A0, n*n, A.data());

        int bufferSize = 0;
        potrf_bufferSize(handle, uplo, n, A.data(), n, &bufferSize);

        buffer.resize(bufferSize);

        potrf(handle, uplo, n, A.data(), n, buffer.data(), bufferSize, info.data());
        CUDA_CHECK_ERROR;

        return 0;
    }

    int solve(R *const b) {
        //ChooseCusolverDnFunc(potrs, Zpotrs, Cpotrs);
        const bool runDoublePrecision = std::is_same<R, double>::value;
        ChooseCusolverDnFunc(potrs, Dpotrs, Spotrs);

        // b will be rewritten in place on return
        potrs(handle, uplo, n, 1, A.data(), n, b, n, info.data());
        return 0;
    }
};

#undef ChooseCusolverDnFunc 
