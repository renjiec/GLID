#pragma once
#include <engine.h>
#include <cstdint>
#include <cstdarg>

#include <vector>
#include <Eigen/Sparse>

#define ENABLE_MATLAB

//#define ensure(cond, msg) if(!cond) fprintf(stderr, "%s", msg);
inline void ensure(bool cond, const char *msg, ...)
{   if (!cond) { va_list args; va_start(args, msg); vfprintf(stderr, (msg+std::string("\n")).c_str(), args); va_end(args); } }

#define ensureTypeMatch(R, m, othertype) ensure(MatlabNum<R>::id == mxGetClassID(m), "Matlab type does not match "##othertype)

class MatlabEngine
{
public:
    bool consoleOutput;

	MatlabEngine():eng(nullptr), consoleOutput(true)	{ }
	virtual ~MatlabEngine() { if(eng) close(); }

	// Run inside matlab: enableservice('AutomationServer', true)
	bool connect(const std::string &dir, bool closeAll=false);

	bool connected() const { return eng!=nullptr; }
    void setEnable(bool v) { if (v) connect(""); else close(); }

	void eval(const std::string &cmd);

	void close();

	void hold_on()	{	eval("hold on;");	}

	void hold_off()	{	eval("hold off;"); }

	const char *output_buffer()	{	return (*engBuffer)?engBuffer:nullptr;	}

	bool hasVar(const std::string &name)
	{
		ensure(connected(), "Not connected to Matlab!");
		mxArray *m = engGetVariable(eng, name.c_str());
        bool r = (m != nullptr);
        mxDestroyArray(m);
        return r;
	}

	mxArray* getVariable(const std::string &name)
	{
		ensure(connected(), "Not connected to Matlab!");
		mxArray *m = engGetVariable(eng, name.c_str());
        ensure(m!=nullptr, "Matlab doesn't have a variable: %s\n", name.c_str());

		return m;
	}

	int putVariable(const std::string &name, const mxArray *m)
	{	ensure(connected(), "Not connected to Matlab!");	return engPutVariable(eng, name.c_str(), m); 	}


private:
	Engine *eng; // Matlab engine
	static const int lenEngBuffer = 1000000;
	char engBuffer[lenEngBuffer]; // engine buffer for outputting strings
};

MatlabEngine& getMatEngine();
// inline void matlabEval(const char* cmd) { getMatEngine().eval(cmd); }
inline void matlabEval(const std::string &cmd) { getMatEngine().eval(cmd.c_str()); }

inline bool matEngineConnected() { return getMatEngine().connected();  }

template<typename R>
struct MatlabNum
{
	static const mxClassID id = mxUNKNOWN_CLASS;
};

template<>	struct MatlabNum<bool>	{	static const mxClassID id = mxLOGICAL_CLASS; };
template<>	struct MatlabNum<char>	{	static const mxClassID id = mxCHAR_CLASS; };
template<>	struct MatlabNum<int>	{	static const mxClassID id = mxINT32_CLASS; };
template<>	struct MatlabNum<float>	{	static const mxClassID id = mxSINGLE_CLASS; };
template<>	struct MatlabNum<double>{	static const mxClassID id = mxDOUBLE_CLASS; };

template<typename R>
inline mxArray* createMatlabArray(const mwSize *dims, int ndim)
{	return mxCreateNumericArray(ndim, dims, MatlabNum<R>::id, mxREAL);	}

////////////////////////////////////////////////////////////////////////////////////////////////////
template<class R>
void VecVec2IdxVec(const std::vector<std::vector<R> >& in, std::vector<int>& v, std::vector<int>& idx)
{
	const size_t nvec = in.size();

	idx.resize(nvec+1);
	idx[0] = 0;

	size_t n = 0;
	for(size_t i=0; i<nvec; i++) n += in[i].size();
	v.clear(); 
	v.reserve(n);

	for(size_t i=0; i<nvec; i++){
		v.insert(v.end(), in[i].begin(), in[i].end());
		idx[i+1] = idx[i]+int(in[i].size());
	}
}

template<class R>
std::vector<std::vector<R> > IdxVec2VecVec(const std::vector<int>& vcat, const std::vector<int>& vidx)
{
	ensure(!vidx.empty(), "empty indices");

	const size_t nvec = vidx.size()-1;
	std::vector<std::vector<R> > v(nvec);

	for(size_t i=0; i<nvec; i++) v[i] = std::vector<R>(vcat.begin()+vidx[i], vcat.begin()+vidx[i+1]);

	return v;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <class R>
void vector2matlab(const std::vector<R> &v, mxArray *m)
{
	ensureTypeMatch(R, m, "std::vector"); 
	R *pm = (R *)mxGetData(m);
	for ( unsigned i = 0 ; i < v.size() ; i++ )
		pm[i] = (R)v[i];
}


template<typename R=double>
void vector2matlab(const std::string &name , const std::vector<R> &v)
{
	mwSize dim[] = { v.size() };
	mxArray *m = createMatlabArray<R>(dim, 1);

	vector2matlab(v, m);
	getMatEngine().putVariable(name, m);
	mxDestroyArray(m);
}

template <class R>
std::vector<R> matlab2vector(const mxArray *m)
{
	ensureTypeMatch(R, m, "std::vector"); 
    if (mxIsSparse(m)){ ensure(false, "matrix is sparse!"); return std::vector<R>(); }

	const R *pm = (R*)mxGetData(m);
	return std::vector<R>( pm, pm + mxGetNumberOfElements(m) );
}

inline std::string matlab2string(const std::string &name)
{
	mxArray *m = getMatEngine().getVariable(name);
    //std::wstring str(mxGetChars(m));
    if (!m) return std::string();

    size_t len = mxGetNumberOfElements(m) + 1;
    std::vector<char> str(len);
    mxGetString(m, str.data(), len);
    mxDestroyArray(m);
    return std::string(str.cbegin(), str.cend()-1);
}

inline std::vector<std::string> matlab2strings(const std::string &name)
{
    mxArray *m = getMatEngine().getVariable(name);
    std::vector<std::string> strs;
    if (!m) return strs;

    for (int i = 0; i < mxGetNumberOfElements(m); i++) {
        mxArray *mstr = mxGetCell(m, i);

        size_t len = mxGetNumberOfElements(mstr) + 1; // for \0 end
        std::vector<char> str(len);
        mxGetString(mstr, str.data(), len);
        strs.push_back(std::string(str.cbegin(), str.cend()-1));
    }

    mxDestroyArray(m);
    return strs;
}

inline void string2matlab(const std::string &name, const std::string &val)
{
    mxArray *m = mxCreateString(val.c_str());
    getMatEngine().putVariable(name, m);
    mxDestroyArray(m);
}


template <class R=double>
std::vector<R> matlab2vector(const std::string &name, bool temp=false)
{
    const std::string tempname("mytempval4c");
    if (temp)
        getMatEngine().eval(tempname + "=" + name + ";");

	mxArray *m = getMatEngine().getVariable(temp?tempname:name);
    if (!m)  return std::vector<R>();

	std::vector<R> v = matlab2vector<R>(m);

	mxDestroyArray(m);

    if (temp)
        getMatEngine().eval( "clear " + tempname + ";");

	return v;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template<class R=double>
void vecvec2matlabcell(const std::string &name, const std::vector<std::vector<R> > &v)
{
	std::vector<int> vcat, vidx;
	VecVec2IdxVec(v, vcat, vidx);
	vector2matlab("vcat_tmp", vcat);
	vector2matlab("vidx_tmp", vidx);

	std::stringstream ss;
	ss<<name<<" = indexedArray2cell(vcat_tmp+1, vidx_tmp+1); clear vcat_tmp vidx_tmp;";

	matlabEval( ss.str() );
}

inline bool incMatCell(const std::string &name)
{
	std::stringstream ss;
	ss<<name<<" = cellfun( @(x) x+1, "<<name<<", 'UniformOutput', false);";
	matlabEval( ss.str() );
}

inline bool decMatCell(const std::string &name)
{
	std::stringstream ss;
	ss<<name<<" = cellfun( @(x) x-1, "<<name<<", 'UniformOutput', false);";
	matlabEval( ss.str() );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template<class R>
void matlabcell2idxvec(const std::string &name, std::vector<R> &vcat, std::vector<R> &vidx)
{
	std::stringstream ss;
	ss<<"[vcat_tmp vidx_tmp]=cell2indexedArray("<<name<<");";

	matlabEval( ss.str() );
	vcat = matlab2vector<int>("vcat_tmp");
	vidx = matlab2vector<int>("vidx_tmp");
	matlabEval( "clear vcat_tmp vidx_tmp;" );
}

template<class R>
std::vector<std::vector<R> > matlabcell2vecvec(const std::string &name)
{
	std::vector<int> vcat, vidx;
	matlabcell2vecvec(name, vcat, vidx);
	return IdxVec2VecVec(vcat, vidx);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
//template <class M>
//void matlab2eigen(const mxArray *m , Eigen::MatrixBase<M> &v)
//{
//	typedef M::Scalar R;
//	const mwSize *dim = mxGetDimensions(m);
//	const R *pm = (R*)mxGetData(m);
//
//	ensure(dim[0]==v.rows() && dim[1]==v.cols());
//	//v.resize(dim[0], dim[1]);
//	 
//	for ( unsigned i = 0 ; i < dim[0] ; i++ )
//		for ( unsigned j = 0 ; j < dim[1] ; j++ ) {
//			const mwSize ind2[] = {i, j};
//			v(i,j) = pm[mxCalcSingleSubscript(m, 2, ind2)];
//		}
//}
//
//template <class M>
//void matlab2eigen(const std::string &name, Eigen::MatrixBase<M> &v)
//{
//	mxArray *m = getMatEngine().getVariable(name);
//	if ( !m ) 	return;
//
//	matlab2eigen(m, v);
//
//	mxDestroyArray(m);
//}

template <class EigenMatrix>
void matlab2eigen(const mxArray *m , EigenMatrix &v)
{
	typedef typename EigenMatrix::Scalar R;
	ensureTypeMatch(R, m, "Eigen::Matrix"); 
    if (mxIsSparse(m)){ ensure(false, "matrix is sparse!"); return; }

	const mwSize *dim = mxGetDimensions(m);

	v = Eigen::Map<const Eigen::Matrix<R,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> >((R*)mxGetData(m), dim[0], dim[1]);
}

template <class Matrix>
void matlab2eigen(const std::string &name, Matrix &v, bool temp=false)
{
    const std::string tempname("mytempval4c");
    if (temp)
        getMatEngine().eval(tempname + "=" + name + ";");

	mxArray *m = getMatEngine().getVariable(temp?tempname:name);
    if (!m)  return;

	matlab2eigen(m, v);

	mxDestroyArray(m);

    if (temp)
        getMatEngine().eval( "clear " + tempname + ";");
}

inline Eigen::MatrixXcd matlab2eigenComplex(const std::string &name)
{
	mxArray *m = getMatEngine().getVariable(name);
    if (!m) return Eigen::MatrixXcd();

    ensure(!mxIsSparse(m), "matrix is sparse!"); 

    const mwSize *dim = mxGetDimensions(m);
    Eigen::MatrixXcd v(dim[0], dim[1]);
    v.real() = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(mxGetPr(m), dim[0], dim[1]);

    if (mxIsComplex(m))
        v.imag() = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(mxGetPi(m), dim[0], dim[1]);
    else
        v.imag().setZero();

	mxDestroyArray(m);
    return v;
}


template<class Mat>
inline void eigen2matlabComplex(const std::string &name, const Mat &vr, const Mat &vi)
{
    mwSize dim[] = { vr.rows(), vr.cols() };
    mxArray *m = mxCreateNumericArray(2, dim, MatlabNum<double>::id, mxCOMPLEX);

    using MapMat = Eigen::Map < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > ;
    MapMat(mxGetPr(m), dim[0], dim[1]) = vr;
    MapMat(mxGetPi(m), dim[0], dim[1]) = vi;

    getMatEngine().putVariable(name, m);
    mxDestroyArray(m);
}

template <class M>
void eigen2matlab(const Eigen::MatrixBase<M> &v, mxArray *m)
{
	typedef M::Scalar R;
	ensureTypeMatch(R, m, "Eigen::Matrix"); 

    using namespace Eigen;
    Map<Matrix<R, Dynamic, Dynamic, ColMajor> >((R*)mxGetData(m), v.rows(), v.cols()) = v;
}

inline void scalar2matlab(const std::string &name, double v) {
    mxArray *m = mxCreateDoubleScalar(v);
    getMatEngine().putVariable(name, m);
	mxDestroyArray(m);
}

inline double matlab2scalar(const std::string &name, double fallback=0, bool temp=false) {
    const std::string tempname("mytempval4c");

    auto &eng = getMatEngine();
    if (temp)  eng.eval(tempname + "=" + name + ";");

    if (!eng.hasVar(temp ? tempname : name)) return fallback;

	mxArray *m = eng.getVariable(temp?tempname:name);
    ensure(mxIsScalar(m), "Matlab: %s is not a scalar!", name.c_str());

    double r = (m && mxIsScalar(m))?mxGetScalar(m):fallback;
    mxDestroyArray(m);

    if (temp)
        eng.eval( "clear " + tempname + ";");

    return r; 
}

template<class M>
void eigen2matlab(const std::string &name, const Eigen::MatrixBase<M> &v)
{
	typedef typename M::Scalar R;
	mwSize dim[] = { (mwSize)v.rows(), (mwSize)v.cols() };
	mxArray *m = createMatlabArray<R>(dim, 2);
	eigen2matlab(v, m);
	getMatEngine().putVariable(name, m);
	mxDestroyArray(m);
}


template <class R>
void eigen2matlab(const std::string &name, Eigen::SparseMatrix<R> &A)
{
	const int n = A.nonZeros();
	MatrixXd sp(n, 3);
	int ind = 0;
	for ( int k = 0 ; k < A.outerSize() ; k++ ) {
		for ( SparseMatrix<R>::InnerIterator it(A,k) ; it ; ++it ) {
			//sp(ind,0) = it.row();
			//sp(ind,1) = it.col();
			sp(ind,0) = it.row()+1;
			sp(ind,1) = it.col()+1;
			sp(ind, 2) = it.value();
			ind++;
		}
	}

	//sp.leftCols<2>() += Eigen::MatrixXd::Ones(n, 2);

	char name2[50];
	sprintf(name2, "%s_", name);
	eigen2matlab(name2, sp);

	char cmd[1000];
	sprintf(cmd, "%s = sparse(%s(:,1), %s(:,2), %s(:,3), %d, %d, %d); clear %s;",
				name, name2, name2, name2, A.rows(), A.cols(), n, name2);
	matlabEval(cmd);
}

template <class R>
void matlab2eigen(const std::string &name, Eigen::SparseMatrix<R> &A)
{
	char cmd[1000];
	sprintf(cmd,
		"[i4Maya, j4Maya, v4Maya] = find(%s);\n"
		"ijv4Maya = [i4Maya-1 j4Maya-1 v4Maya];\n"
		"sz4Maya = int32(size(%s));\n"
		"clear i4Maya j4Maya v4Maya;",
		name, name);

	matlabEval(cmd);

	Eigen::RowVector2i sz;
	matlab2eigen("sz4Maya", sz);

	MatrixXd ijv;
	matlab2eigen("ijv4Maya", ijv);

	matlabEval("clear sz4Maya ijv4Maya;");

	const int m = sz(0), n = sz(1);
	A = SparseMatrix<R>(m, n);
	A.reserve(ijv.rows());
	for ( int i = 0 ; i < ijv.rows() ; i++ ) {
		int k = int( ijv(i,1) );
		if ( i == 0 || k != ijv(i-1, 1) )
			A.startVec(k);
		A.insertBack( int(ijv(i,0)), k) = R( ijv(i, 2) );
	}
	A.finalize();
}

