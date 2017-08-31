#ifndef GLPROGRAM_H
#define GLPROGRAM_H

#include <string>
#include <map>

#define OPENGL_VERSION 3

class GLProgram
{
public:
    enum ShaderType { VERTEX_SHADER = 0, GEOMETRY_SHADER, FRAGMENT_SHADER, TESS_CONTROL_SHADER, TESS_EVALUATION_SHADER, COMPUTE_SHADER, NUM_SHADER_TYPE };
    static const char *const ShaderTypeStrs[NUM_SHADER_TYPE];

private:
    unsigned int  pProgram;
    unsigned int  pShader[NUM_SHADER_TYPE];
    std::string logString;
    bool          rowmajor; // for setting matrix uniforms

    struct Uniform { unsigned int type; int location, size, stride; };

    struct BlockUniform { unsigned int type; int offset, size, arrayStride; };

	/// stores information for a block and its uniforms
    struct UniformBlock {
        /// size of the uniform block
        int size;
        /// buffer bound to the index point
        unsigned int buffer;
        /// binding index
        unsigned int bindingIndex;
        /// uniforms information
        std::map<std::string, BlockUniform > uniformOffsets;
    };

    std::map < std::string, Uniform > pUniforms;
    std::map < std::string, UniformBlock > pBlocks;

public:
    GLProgram();
    ~GLProgram();

    void reset();

    bool   compileShaderFromFile( const char * fileName, ShaderType type );
    bool   compileShaderFromString( const std::string & source, ShaderType type );
	int    compileAndLinkAllShadersFromString(const std::string &vtxShaderStr, const std::string &fragShaderStr, const std::string &geomShaderStr="");
	int    compileAndLinkAllShadersFromFile(const char * vertexShaderFile, const char * fragShaderFile, const char *geomShaderFile=NULL);

    std::string getShaderInfoLog(ShaderType type);
    std::string getProgramInfoLog();
    std::string getAllInfoLog();

    bool   link();
    bool   validate();
    void   bind();
    void   unbind();

	bool   isbinded();

    std::string log() const { return logString; }

    int    getHandle() { return pProgram; }

    bool   shaderCompiled(ShaderType);
    bool   linked();

    void   cacheUniforms();
    void   cacheBlocks();

    void   setUniformPVOID(const char *, void *);

    // only work on OpenGL4 hardware
#if OPENGL_VERSION >= 4
    void   setUniformPVOID_GL4(const char *, void *);
#endif

    void   setBlock(const char *, void *);
    void   setBlockUniform(const char*, const char*, void *);
    void   setBlockUniformArrayElement(const char *, const char *, int, void *);

    template<typename R, typename... Rs>
    typename std::enable_if<!std::is_pointer<R>::value&&std::is_scalar<R>::value,void>::type setUniform(const char *name, R v1, Rs... v) {
        static_assert(sizeof(R) == 4 || sizeof(R) == 8, "OpenGL only takes data of unsigned, int, float, double");
        R vals[] = { v1, v... };
        setUniformPVOID(name, (void*)vals);
    }

    template<typename R>
    typename std::enable_if<std::is_scalar<R>::value,void>::type setUniform(const char *name, R* vals) {
        static_assert(sizeof(R) == 4 || sizeof(R) == 8, "OpenGL only takes data of unsigned, int, float, double");
        setUniformPVOID(name, (void*)vals);
    }

    void   bindAttribLocation(unsigned int location, const char * name);
    void   bindFragDataLocation( unsigned int location, const char * name );

    int    getAttribLocation(const char * name);

    void   printActiveUniforms();
    void   printActiveAttribs();
};

#endif // GLPROGRAM_H
