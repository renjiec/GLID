#pragma once

#include <vector>

#ifdef _DEBUG
#include <cassert>
#define MAKESURE(x) assert(x)
#else
#define MAKESURE(x) 
#endif

template<typename R>
struct GLType {
    //static const GLenum val;
    //int Must_Use_Specialization[-1];
    //static_assert(false, "undefined type for OpenGL");
};

template<>	struct GLType<bool>	{	static const GLenum val = GL_BOOL; };
template<>	struct GLType<int>	{	static const GLenum val = GL_INT; };
template<>	struct GLType<float>	{	static const GLenum val = GL_FLOAT; };
template<>	struct GLType<double>	{	static const GLenum val = GL_DOUBLE; };

template<typename R, int dim, bool isElementArray=false>
class GLArray
{
public:
    typename R Scalar;

    static const GLenum GLBufferTarget = isElementArray?GL_ELEMENT_ARRAY_BUFFER:GL_ARRAY_BUFFER;
    static_assert( (!isElementArray) || (isElementArray && (std::is_same<R,int>::value || std::is_same<R,unsigned int>::value)), "the numerical type of an OpenGL element array should be (unsigned) int");

private:
    unsigned int handle;

public:
    GLArray(const R *v = nullptr, int n = 0) :handle(0) { if (v) uploadData(v, n); }
    GLArray(GLArray&& t)  { std::swap(handle, t.handle); }
    GLArray& operator=(GLArray&& t) { std::swap(handle, t.handle); return *this; };
    GLArray(const GLArray&) = delete;
    GLArray& operator=(const GLArray&) = delete;

    ~GLArray() { if(handle) glDeleteBuffers(1, &handle); }

    //int dimension() const { return dim; }
    void bind() { MAKESURE(glIsBuffer(handle)); glBindBuffer(GLBufferTarget, handle); }
	void unbind()   { glBindBuffer(GLBufferTarget, 0); }
	
    bool empty() const { return handle == 0; }

    void allocateStorage(int n)
    {
        // make sure OpenGL Context is created before this!
        if (handle && size() == n) return;
        if (!handle)  glGenBuffers(1, &handle);

        glBindBuffer(GLBufferTarget, handle);
        glBufferData(GLBufferTarget, (dim * n) * sizeof(R), nullptr, GL_STATIC_DRAW);
    }

    void uploadData(const R *v, int n, int offset = 0, bool allocateIfNeeded = true)
    {
        if (!offset && allocateIfNeeded)    allocateStorage(n);
        bind();
        glBufferSubData(GLBufferTarget, (dim*offset)*sizeof(R), (dim * n) * sizeof(R), v);
    }

    void downloadData(R* v, int n, int offset = 0)
    {
        bind();
        glGetBufferSubData(GLBufferTarget, (dim*offset)*sizeof(R), (dim * n) * sizeof(R), v);
    }

    void at(int i,  R *x)
    {
        bind();
		glGetBufferSubData(GLBufferTarget, i*dim*sizeof(R), dim* sizeof(R), x);
    }

    void setAt(int i, const R *x)
    {
        bind();
		glBufferSubData(GLBufferTarget, i*dim*sizeof(R), dim* sizeof(R), x);
    }

    int size() 
    {
        bind();
        int n;
        glGetBufferParameteriv(GLBufferTarget, GL_BUFFER_SIZE, &n);
        return n / dim / sizeof(R);
    }

};

