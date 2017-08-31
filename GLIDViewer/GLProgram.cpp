#include "GLProgram.h"

#include <fstream>
#include <sstream>

#include <cassert>
#include <cstring>
#include <sys/stat.h>


#if OPENGL_VERSION >= 4
#include "gl_core_4_4.h"
#else
#include "gl_core_3_3.h"
#endif

bool fileExists(const std::string & fileName)
{
    struct stat info;
    return 0 == stat(fileName.c_str(), &info);
}

std::string textFileRead(const std::string & fileName)
{
    std::ifstream inFile( fileName, std::ios::in );

    std::ostringstream code;
    while (inFile.good()) {
        int c = inFile.get();
        if (!inFile.eof())
            code << (char)c;
    }
    inFile.close();

    return code.str();
}

int typeSize(int type) 
{
	switch(type) {
		// Floats
		case GL_FLOAT: 	    		return sizeof(float);
        case GL_FLOAT_VEC2:  		return sizeof(float) * 2;
        case GL_FLOAT_VEC3:  		return sizeof(float) * 3;
        case GL_FLOAT_VEC4:  		return sizeof(float) * 4;

		// Doubles
		case GL_DOUBLE: 			return sizeof(double);
#if OPENGL_VERSION >= 4
		case GL_DOUBLE_VEC2:    	return sizeof(double) * 2;
		case GL_DOUBLE_VEC3:  		return sizeof(double) * 3;
		case GL_DOUBLE_VEC4:  		return sizeof(double) * 4;
#endif

		// Samplers, Ints and Bools
		case GL_SAMPLER_1D:
		case GL_SAMPLER_2D:
		case GL_SAMPLER_3D:
		case GL_SAMPLER_CUBE:
		case GL_SAMPLER_1D_SHADOW:
		case GL_SAMPLER_2D_SHADOW:
		case GL_SAMPLER_1D_ARRAY:
		case GL_SAMPLER_2D_ARRAY:
		case GL_SAMPLER_1D_ARRAY_SHADOW:
		case GL_SAMPLER_2D_ARRAY_SHADOW:
		case GL_SAMPLER_2D_MULTISAMPLE:
		case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
		case GL_SAMPLER_CUBE_SHADOW:
		case GL_SAMPLER_BUFFER:
		case GL_SAMPLER_2D_RECT:
		case GL_SAMPLER_2D_RECT_SHADOW:
		case GL_INT_SAMPLER_1D:
		case GL_INT_SAMPLER_2D:
		case GL_INT_SAMPLER_3D:
		case GL_INT_SAMPLER_CUBE:
		case GL_INT_SAMPLER_1D_ARRAY:
		case GL_INT_SAMPLER_2D_ARRAY:
		case GL_INT_SAMPLER_2D_MULTISAMPLE:
		case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
		case GL_INT_SAMPLER_BUFFER:
		case GL_INT_SAMPLER_2D_RECT:
		case GL_UNSIGNED_INT_SAMPLER_1D:
		case GL_UNSIGNED_INT_SAMPLER_2D:
		case GL_UNSIGNED_INT_SAMPLER_3D:
		case GL_UNSIGNED_INT_SAMPLER_CUBE:
		case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
		case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
		case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
		case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
		case GL_UNSIGNED_INT_SAMPLER_BUFFER:
		case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
		case GL_BOOL:  
		case GL_INT : 
			return sizeof(int);
		case GL_BOOL_VEC2:
		case GL_INT_VEC2:  
			return sizeof(int) * 2;
		case GL_BOOL_VEC3:
		case GL_INT_VEC3:  
			return sizeof(int) * 3;
		case GL_BOOL_VEC4:
		case GL_INT_VEC4:  
			return sizeof(int) * 4;

		// Unsigned ints
        case GL_UNSIGNED_INT: 		return sizeof(unsigned int);
        case GL_UNSIGNED_INT_VEC2:  return sizeof(unsigned int) * 2;
		case GL_UNSIGNED_INT_VEC3:  return sizeof(unsigned int) * 3;
		case GL_UNSIGNED_INT_VEC4:  return sizeof(unsigned int) * 4;

		// Float Matrices
		case GL_FLOAT_MAT2:			return sizeof(float) * 4;
		case GL_FLOAT_MAT3:			return sizeof(float) * 9;
		case GL_FLOAT_MAT4:			return sizeof(float) * 16;
		case GL_FLOAT_MAT2x3:		return sizeof(float) * 6;
		case GL_FLOAT_MAT2x4:		return sizeof(float) * 8;
		case GL_FLOAT_MAT3x2:		return sizeof(float) * 6;
		case GL_FLOAT_MAT3x4:		return sizeof(float) * 12;
        case GL_FLOAT_MAT4x2:		return sizeof(float) * 8;
        case GL_FLOAT_MAT4x3:		return sizeof(float) * 12;

		// Double Matrices
#if OPENGL_VERSION >= 4
		case GL_DOUBLE_MAT2:		return sizeof(double) * 4;
		case GL_DOUBLE_MAT3:		return sizeof(double) * 9;
		case GL_DOUBLE_MAT4:		return sizeof(double) * 16;
		case GL_DOUBLE_MAT2x3:		return sizeof(double) * 6;
		case GL_DOUBLE_MAT2x4:		return sizeof(double) * 8;
		case GL_DOUBLE_MAT3x2:		return sizeof(double) * 6;
		case GL_DOUBLE_MAT3x4:		return sizeof(double) * 12;
		case GL_DOUBLE_MAT4x2:		return sizeof(double) * 8;
		case GL_DOUBLE_MAT4x3:		return sizeof(double) * 12;
#endif
		default: return 0;
	}
}


const char *const GLProgram::ShaderTypeStrs[NUM_SHADER_TYPE] = { "Vertex Shader", "Geometry Shader", "Tesselation Control Shader", "Tesselation Evaluation Shader", "Fragment Shader", "Compute Shader" };

GLProgram::GLProgram() : pProgram(0), rowmajor(true) { 
    std::fill_n(pShader, int(NUM_SHADER_TYPE), 0);
}

GLProgram::~GLProgram() { reset(); }

void GLProgram::reset()
{
    if (pProgram){
        glDeleteProgram(pProgram);
        pProgram = 0;
    }

    for (int i = 0; i < NUM_SHADER_TYPE; i++)
        if (pShader[i]){
            //glDetachShader(progHandle, shaderHandles[i]);
            glDeleteShader(pShader[i]);
            pShader[i] = 0;
        }
}

int GLProgram::compileAndLinkAllShadersFromString(const std::string &vtxShaderStr, const std::string &fragShaderStr, const std::string &geomShaderStr)
{
    //reset();
    if (!compileShaderFromString(vtxShaderStr, VERTEX_SHADER)){
		fprintf(stderr, "Vertex shader failed to compile!\n%s", log().c_str());
        return -1;
    }

    if (!compileShaderFromString(fragShaderStr, FRAGMENT_SHADER)){
		fprintf(stderr, "Fragment shader failed to compile!\n%s", log().c_str());
        return -1;
    }


    if (!geomShaderStr.empty() && !compileShaderFromString(geomShaderStr, GEOMETRY_SHADER)){
		fprintf(stderr, "Geometry shader failed to compile!\n%s", log().c_str());
        return -1;
    }

	if (!link()){
		fprintf(stderr, "Shader program failed to link!\n%s", log().c_str());
		return -1;
	}

	return 0;
}

int GLProgram::compileAndLinkAllShadersFromFile(const char *vertexShaderFile, const char *fragShaderFile, const char *geomShaderFile)
{
    return compileAndLinkAllShadersFromString(textFileRead(vertexShaderFile), textFileRead(fragShaderFile), geomShaderFile ? textFileRead(geomShaderFile) : std::string());
}

bool GLProgram::compileShaderFromFile(const char * fileName, ShaderType type)
{
    if (!fileExists(fileName)) {
        logString = "Shader File: " + std::string(fileName) + " not found.";
        return false;
    }

    return compileShaderFromString(textFileRead(fileName), type);
}

bool GLProgram::compileShaderFromString(const std::string & source, ShaderType type)
{
    const GLenum GLShaderTypes[NUM_SHADER_TYPE] = {
        GL_VERTEX_SHADER,
        GL_GEOMETRY_SHADER,
        GL_FRAGMENT_SHADER,
#if OPENGL_VERSION >= 4
        GL_TESS_CONTROL_SHADER,
        GL_TESS_EVALUATION_SHADER,
        GL_COMPUTE_SHADER
#endif
    };

    GLuint &hdl = pShader[type];
    if (!hdl) hdl = glCreateShader(GLShaderTypes[type]);

    const char * ss = source.c_str();
    glShaderSource(hdl, 1, &ss, NULL);

    // Compile the shader
    glCompileShader(hdl);

    // Check for errors
    int result;
    glGetShaderiv(hdl, GL_COMPILE_STATUS, &result);
    if (GL_FALSE == result) {
        // Compile failed, store log and return false
        logString = getShaderInfoLog(type);
        return false;
    }
    
    // update uniform cache if program is already linked
    if (linked()) { cacheUniforms(); cacheBlocks(); }
    return true;
}

bool GLProgram::link()
{
    if (!pProgram) {
        pProgram = glCreateProgram();
        if (pProgram == 0) {
            logString = "Unable to create shader program.";
            return false;
        }

        // attach shaders
        for (int i = 0; i < NUM_SHADER_TYPE; i++)
            if (pShader[i]>0) glAttachShader(pProgram, pShader[i]);
    }

    glLinkProgram(pProgram);

    if (!linked()){
        // fail to link, Store log and return false
        logString = getProgramInfoLog();
        return false;
    }

    // cache all uniforms
    cacheUniforms();
    cacheBlocks();
    return true;
}


void GLProgram::cacheUniforms() 
{
    pUniforms.clear();

    int count;
    glGetProgramiv(pProgram, GL_ACTIVE_UNIFORMS, &count);

    int maxUniLength;
    glGetProgramiv(pProgram, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxUniLength);

    char *name = new char[maxUniLength];
    for (int i = 0; i < count; ++i) {
        GLsizei actualLen;
        GLenum type;
        GLint size;
        glGetActiveUniform(pProgram, i, maxUniLength, &actualLen, &size, &type, name);

        size_t namelen = strlen(name);
        if (namelen>3 && !strcmp(name + namelen - 3, "[0]")) {
            name[namelen - 3] = '\0';
        }

        // -1 indicates that is not an active uniform, although it may be present in a uniform block
        int loc = glGetUniformLocation(pProgram, name);
        if (loc != -1) {
            GLint uniArrayStride;
            glGetActiveUniformsiv(pProgram, 1, (GLuint*)&i, GL_UNIFORM_ARRAY_STRIDE, &uniArrayStride);
            pUniforms[name] = { type, loc, size, uniArrayStride };
        }
    }
    delete[] name;
}

void GLProgram::cacheBlocks() 
{
    pBlocks.clear();

    int count;
	glGetProgramiv(pProgram, GL_ACTIVE_UNIFORM_BLOCKS, &count);
    if (count == 0) return;

    int maxUniBlkLength;
    glGetProgramiv(pProgram, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &maxUniBlkLength);

    char *name = new char[maxUniBlkLength];

    int maxUniLength;
    glGetProgramiv(pProgram, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxUniLength);
    char *name2 = new char[maxUniLength];

	int dataSize, actualLen, activeUnif;
	int uniType, uniSize, uniOffset, uniMatStride, uniArrayStride, auxSize;

    int blockCount = 0;
	UniformBlock block;
	for (int i = 0; i < count; ++i) {
		// Get buffers name
		glGetActiveUniformBlockName(pProgram, i, maxUniBlkLength, &actualLen, name);

        bool newBlock = true;
        if (pBlocks.count(name)) {
			newBlock = false;
			block = pBlocks[name];
		}

        blockCount += newBlock;

		/*if (!pBlocks.count(name))*/ {
			// Get buffers size
			glGetActiveUniformBlockiv(pProgram, i, GL_UNIFORM_BLOCK_DATA_SIZE, &dataSize);

			if (newBlock) {
				glGenBuffers(1, &block.buffer);
				glBindBuffer(GL_UNIFORM_BUFFER, block.buffer);
				glBufferData(GL_UNIFORM_BUFFER, dataSize, NULL, GL_DYNAMIC_DRAW);
				glUniformBlockBinding(pProgram, i, blockCount);
				glBindBufferRange(GL_UNIFORM_BUFFER, blockCount, block.buffer, 0, dataSize);
			}
			else {
				glBindBuffer(GL_UNIFORM_BUFFER, block.buffer);
				glUniformBlockBinding(pProgram, i, block.bindingIndex);
			}
			glGetActiveUniformBlockiv(pProgram, i, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &activeUnif);

			unsigned int *indices = new unsigned int[activeUnif];
			glGetActiveUniformBlockiv(pProgram, i, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, (int *)indices);
			
			for (int k = 0; k < activeUnif; ++k) {
				glGetActiveUniformName(pProgram, indices[k], maxUniLength, &actualLen, name2);
				glGetActiveUniformsiv(pProgram, 1, &indices[k], GL_UNIFORM_TYPE, &uniType);
				glGetActiveUniformsiv(pProgram, 1, &indices[k], GL_UNIFORM_SIZE, &uniSize);
				glGetActiveUniformsiv(pProgram, 1, &indices[k], GL_UNIFORM_OFFSET, &uniOffset);
				glGetActiveUniformsiv(pProgram, 1, &indices[k], GL_UNIFORM_MATRIX_STRIDE, &uniMatStride);
				glGetActiveUniformsiv(pProgram, 1, &indices[k], GL_UNIFORM_ARRAY_STRIDE, &uniArrayStride);
			
				if (uniArrayStride > 0)
					auxSize = uniArrayStride * uniSize;
				
				else if (uniMatStride > 0) {
					switch(uniType) {
						case GL_FLOAT_MAT2:
						case GL_FLOAT_MAT2x3:
						case GL_FLOAT_MAT2x4:
#if OPENGL_VERSION >= 4
						case GL_DOUBLE_MAT2:
						case GL_DOUBLE_MAT2x3:
						case GL_DOUBLE_MAT2x4:
#endif
							auxSize = 2 * uniMatStride;
							break;
						case GL_FLOAT_MAT3:
						case GL_FLOAT_MAT3x2:
						case GL_FLOAT_MAT3x4:
#if OPENGL_VERSION >= 4
						case GL_DOUBLE_MAT3:
						case GL_DOUBLE_MAT3x2:
						case GL_DOUBLE_MAT3x4:
#endif
							auxSize = 3 * uniMatStride;
							break;
						case GL_FLOAT_MAT4:
						case GL_FLOAT_MAT4x2:
						case GL_FLOAT_MAT4x3:
#if OPENGL_VERSION >= 4
						case GL_DOUBLE_MAT4:
						case GL_DOUBLE_MAT4x2:
						case GL_DOUBLE_MAT4x3:
#endif
							auxSize = 4 * uniMatStride;
							break;
					}
				}
				else
					auxSize = typeSize(uniType);

				block.uniformOffsets[name2] = { (unsigned)uniType, uniOffset, auxSize, uniArrayStride };
			}

            delete[] indices;

			if (newBlock) {
                block.size = dataSize;
                block.bindingIndex = blockCount;
			}
            pBlocks[name] = block;
		}
		//else
		//	glUniformBlockBinding(pProgram, i, pBlocks[name].bindingIndex);

	}
    delete []name2;
    delete []name;
}

void GLProgram::setUniformPVOID(const char *name, void *value)
{
    assert(isbinded());

    auto it = pUniforms.find(name);
    if (it == pUniforms.end()){
        fprintf(stderr, "Uniform: %s not found.\n", name);
        return;
    }

    const Uniform &u = it->second;
    switch (u.type) {
        // Floats
    case GL_FLOAT:
        glUniform1fv(u.location, u.size, (const GLfloat *)value);
        break;
    case GL_FLOAT_VEC2:
        glUniform2fv(u.location, u.size, (const GLfloat *)value);
        break;
    case GL_FLOAT_VEC3:
        glUniform3fv(u.location, u.size, (const GLfloat *)value);
        break;
    case GL_FLOAT_VEC4:
        glUniform4fv(u.location, u.size, (const GLfloat *)value);
        break;

        // Doubles
#if OPENGL_VERSION >= 4
    case GL_DOUBLE:
        glUniform1dv(u.location, u.size, (const GLdouble *)value);
        break;
    case GL_DOUBLE_VEC2:
        glUniform2dv(u.location, u.size, (const GLdouble *)value);
        break;
    case GL_DOUBLE_VEC3:
        glUniform3dv(u.location, u.size, (const GLdouble *)value);
        break;
    case GL_DOUBLE_VEC4:
        glUniform4dv(u.location, u.size, (const GLdouble *)value);
        break;
#endif

        // Samplers, Ints and Bools
#if OPENGL_VERSION >= 4
    case GL_IMAGE_1D:
    case GL_IMAGE_2D:
    case GL_IMAGE_3D:
    case GL_IMAGE_2D_RECT:
    case GL_IMAGE_CUBE:
    case GL_IMAGE_BUFFER:
    case GL_IMAGE_1D_ARRAY:
    case GL_IMAGE_2D_ARRAY:
    case GL_IMAGE_CUBE_MAP_ARRAY:
    case GL_IMAGE_2D_MULTISAMPLE:
    case GL_IMAGE_2D_MULTISAMPLE_ARRAY:
    case GL_INT_IMAGE_1D:
    case GL_INT_IMAGE_2D:
    case GL_INT_IMAGE_3D:
    case GL_INT_IMAGE_2D_RECT:
    case GL_INT_IMAGE_CUBE:
    case GL_INT_IMAGE_BUFFER:
    case GL_INT_IMAGE_1D_ARRAY:
    case GL_INT_IMAGE_2D_ARRAY:
    case GL_INT_IMAGE_CUBE_MAP_ARRAY:
    case GL_INT_IMAGE_2D_MULTISAMPLE:
    case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_1D:
    case GL_UNSIGNED_INT_IMAGE_2D:
    case GL_UNSIGNED_INT_IMAGE_3D:
    case GL_UNSIGNED_INT_IMAGE_2D_RECT:
    case GL_UNSIGNED_INT_IMAGE_CUBE:
    case GL_UNSIGNED_INT_IMAGE_BUFFER:
    case GL_UNSIGNED_INT_IMAGE_1D_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE:
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
#endif
    case GL_SAMPLER_1D:
    case GL_SAMPLER_2D:
    case GL_SAMPLER_3D:
    case GL_SAMPLER_CUBE:
    case GL_SAMPLER_1D_SHADOW:
    case GL_SAMPLER_2D_SHADOW:
    case GL_SAMPLER_1D_ARRAY:
    case GL_SAMPLER_2D_ARRAY:
    case GL_SAMPLER_1D_ARRAY_SHADOW:
    case GL_SAMPLER_2D_ARRAY_SHADOW:
    case GL_SAMPLER_2D_MULTISAMPLE:
    case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_SAMPLER_CUBE_SHADOW:
    case GL_SAMPLER_BUFFER:
    case GL_SAMPLER_2D_RECT:
    case GL_SAMPLER_2D_RECT_SHADOW:
    case GL_INT_SAMPLER_1D:
    case GL_INT_SAMPLER_2D:
    case GL_INT_SAMPLER_3D:
    case GL_INT_SAMPLER_CUBE:
    case GL_INT_SAMPLER_1D_ARRAY:
    case GL_INT_SAMPLER_2D_ARRAY:
    case GL_INT_SAMPLER_2D_MULTISAMPLE:
    case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_INT_SAMPLER_BUFFER:
    case GL_INT_SAMPLER_2D_RECT:
    case GL_UNSIGNED_INT_SAMPLER_1D:
    case GL_UNSIGNED_INT_SAMPLER_2D:
    case GL_UNSIGNED_INT_SAMPLER_3D:
    case GL_UNSIGNED_INT_SAMPLER_CUBE:
    case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_BUFFER:
    case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
    case GL_BOOL:
    case GL_INT:
        glUniform1iv(u.location, u.size, (const GLint *)value);
        break;
    case GL_BOOL_VEC2:
    case GL_INT_VEC2:
        glUniform2iv(u.location, u.size, (const GLint *)value);
        break;
    case GL_BOOL_VEC3:
    case GL_INT_VEC3:
        glUniform3iv(u.location, u.size, (const GLint *)value);
        break;
    case GL_BOOL_VEC4:
    case GL_INT_VEC4:
        glUniform4iv(u.location, u.size, (const GLint *)value);
        break;

        // Unsigned ints
    case GL_UNSIGNED_INT:
        glUniform1uiv(u.location, u.size, (const GLuint *)value);
        break;
    case GL_UNSIGNED_INT_VEC2:
        glUniform2uiv(u.location, u.size, (const GLuint *)value);
        break;
    case GL_UNSIGNED_INT_VEC3:
        glUniform3uiv(u.location, u.size, (const GLuint *)value);
        break;
    case GL_UNSIGNED_INT_VEC4:
        glUniform4uiv(u.location, u.size, (const GLuint *)value);
        break;

        // Float Matrices
    case GL_FLOAT_MAT2:
        glUniformMatrix2fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT3:
        glUniformMatrix3fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT4:
        glUniformMatrix4fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT2x3:
        glUniformMatrix2x3fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT2x4:
        glUniformMatrix2x4fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT3x2:
        glUniformMatrix3x2fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT3x4:
        glUniformMatrix3x4fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT4x2:
        glUniformMatrix4x2fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT4x3:
        glUniformMatrix4x3fv(u.location, u.size, rowmajor, (const GLfloat *)value);
        break;

#if OPENGL_VERSION >= 4
        // Double Matrices
    case GL_DOUBLE_MAT2:
        glUniformMatrix2dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT3:
        glUniformMatrix3dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT4:
        glUniformMatrix4dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT2x3:
        glUniformMatrix2x3dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT2x4:
        glUniformMatrix2x4dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT3x2:
        glUniformMatrix3x2dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT3x4:
        glUniformMatrix3x4dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT4x2:
        glUniformMatrix4x2dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT4x3:
        glUniformMatrix4x3dv(u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
#endif
    }

}

#if OPENGL_VERSION >= 4
void GLProgram::setUniformPVOID_GL4(const char *name, void *value)
{
    auto it = pUniforms.find(name);
    if (it == pUniforms.end()){
        fprintf(stderr, "Uniform: %s not found.\n", name);
        return;
    }

    const Uniform &u = it->second;
    switch (u.type) {
        // Floats
    case GL_FLOAT:
        glProgramUniform1fv(pProgram, u.location, u.size, (const GLfloat *)value);
        break;
    case GL_FLOAT_VEC2:
        glProgramUniform2fv(pProgram, u.location, u.size, (const GLfloat *)value);
        break;
    case GL_FLOAT_VEC3:
        glProgramUniform3fv(pProgram, u.location, u.size, (const GLfloat *)value);
        break;
    case GL_FLOAT_VEC4:
        glProgramUniform4fv(pProgram, u.location, u.size, (const GLfloat *)value);
        break;

        // Doubles
    case GL_DOUBLE:
        glProgramUniform1dv(pProgram, u.location, u.size, (const GLdouble *)value);
        break;
    case GL_DOUBLE_VEC2:
        glProgramUniform2dv(pProgram, u.location, u.size, (const GLdouble *)value);
        break;
    case GL_DOUBLE_VEC3:
        glProgramUniform3dv(pProgram, u.location, u.size, (const GLdouble *)value);
        break;
    case GL_DOUBLE_VEC4:
        glProgramUniform4dv(pProgram, u.location, u.size, (const GLdouble *)value);
        break;

        // Samplers, Ints and Bools
    case GL_IMAGE_1D:
    case GL_IMAGE_2D:
    case GL_IMAGE_3D:
    case GL_IMAGE_2D_RECT:
    case GL_IMAGE_CUBE:
    case GL_IMAGE_BUFFER:
    case GL_IMAGE_1D_ARRAY:
    case GL_IMAGE_2D_ARRAY:
    case GL_IMAGE_CUBE_MAP_ARRAY:
    case GL_IMAGE_2D_MULTISAMPLE:
    case GL_IMAGE_2D_MULTISAMPLE_ARRAY:
    case GL_INT_IMAGE_1D:
    case GL_INT_IMAGE_2D:
    case GL_INT_IMAGE_3D:
    case GL_INT_IMAGE_2D_RECT:
    case GL_INT_IMAGE_CUBE:
    case GL_INT_IMAGE_BUFFER:
    case GL_INT_IMAGE_1D_ARRAY:
    case GL_INT_IMAGE_2D_ARRAY:
    case GL_INT_IMAGE_CUBE_MAP_ARRAY:
    case GL_INT_IMAGE_2D_MULTISAMPLE:
    case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_1D:
    case GL_UNSIGNED_INT_IMAGE_2D:
    case GL_UNSIGNED_INT_IMAGE_3D:
    case GL_UNSIGNED_INT_IMAGE_2D_RECT:
    case GL_UNSIGNED_INT_IMAGE_CUBE:
    case GL_UNSIGNED_INT_IMAGE_BUFFER:
    case GL_UNSIGNED_INT_IMAGE_1D_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE:
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
    case GL_SAMPLER_1D:
    case GL_SAMPLER_2D:
    case GL_SAMPLER_3D:
    case GL_SAMPLER_CUBE:
    case GL_SAMPLER_1D_SHADOW:
    case GL_SAMPLER_2D_SHADOW:
    case GL_SAMPLER_1D_ARRAY:
    case GL_SAMPLER_2D_ARRAY:
    case GL_SAMPLER_1D_ARRAY_SHADOW:
    case GL_SAMPLER_2D_ARRAY_SHADOW:
    case GL_SAMPLER_2D_MULTISAMPLE:
    case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_SAMPLER_CUBE_SHADOW:
    case GL_SAMPLER_BUFFER:
    case GL_SAMPLER_2D_RECT:
    case GL_SAMPLER_2D_RECT_SHADOW:
    case GL_INT_SAMPLER_1D:
    case GL_INT_SAMPLER_2D:
    case GL_INT_SAMPLER_3D:
    case GL_INT_SAMPLER_CUBE:
    case GL_INT_SAMPLER_1D_ARRAY:
    case GL_INT_SAMPLER_2D_ARRAY:
    case GL_INT_SAMPLER_2D_MULTISAMPLE:
    case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_INT_SAMPLER_BUFFER:
    case GL_INT_SAMPLER_2D_RECT:
    case GL_UNSIGNED_INT_SAMPLER_1D:
    case GL_UNSIGNED_INT_SAMPLER_2D:
    case GL_UNSIGNED_INT_SAMPLER_3D:
    case GL_UNSIGNED_INT_SAMPLER_CUBE:
    case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_BUFFER:
    case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
    case GL_BOOL:
    case GL_INT:
        glProgramUniform1iv(pProgram, u.location, u.size, (const GLint *)value);
        break;
    case GL_BOOL_VEC2:
    case GL_INT_VEC2:
        glProgramUniform2iv(pProgram, u.location, u.size, (const GLint *)value);
        break;
    case GL_BOOL_VEC3:
    case GL_INT_VEC3:
        glProgramUniform3iv(pProgram, u.location, u.size, (const GLint *)value);
        break;
    case GL_BOOL_VEC4:
    case GL_INT_VEC4:
        glProgramUniform4iv(pProgram, u.location, u.size, (const GLint *)value);
        break;

        // Unsigned ints
    case GL_UNSIGNED_INT:
        glProgramUniform1uiv(pProgram, u.location, u.size, (const GLuint *)value);
        break;
    case GL_UNSIGNED_INT_VEC2:
        glProgramUniform2uiv(pProgram, u.location, u.size, (const GLuint *)value);
        break;
    case GL_UNSIGNED_INT_VEC3:
        glProgramUniform3uiv(pProgram, u.location, u.size, (const GLuint *)value);
        break;
    case GL_UNSIGNED_INT_VEC4:
        glProgramUniform4uiv(pProgram, u.location, u.size, (const GLuint *)value);
        break;

        // Float Matrices
    case GL_FLOAT_MAT2:
        glProgramUniformMatrix2fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT3:
        glProgramUniformMatrix3fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT4:
        glProgramUniformMatrix4fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT2x3:
        glProgramUniformMatrix2x3fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT2x4:
        glProgramUniformMatrix2x4fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT3x2:
        glProgramUniformMatrix3x2fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT3x4:
        glProgramUniformMatrix3x4fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT4x2:
        glProgramUniformMatrix4x2fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;
    case GL_FLOAT_MAT4x3:
        glProgramUniformMatrix4x3fv(pProgram, u.location, u.size, rowmajor, (const GLfloat *)value);
        break;

        // Double Matrices
    case GL_DOUBLE_MAT2:
        glProgramUniformMatrix2dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT3:
        glProgramUniformMatrix3dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT4:
        glProgramUniformMatrix4dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT2x3:
        glProgramUniformMatrix2x3dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT2x4:
        glProgramUniformMatrix2x4dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT3x2:
        glProgramUniformMatrix3x2dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT3x4:
        glProgramUniformMatrix3x4dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT4x2:
        glProgramUniformMatrix4x2dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    case GL_DOUBLE_MAT4x3:
        glProgramUniformMatrix4x3dv(pProgram, u.location, u.size, rowmajor, (const GLdouble *)value);
        break;
    }
}
#endif

void GLProgram::setBlock(const char* name, void *value) {

	if (pBlocks.count(name) != 0) {
		glBindBuffer(GL_UNIFORM_BUFFER, pBlocks[name].buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, pBlocks[name].size, value);
		glBindBuffer(GL_UNIFORM_BUFFER,0);
	}
}


void GLProgram::setBlockUniform(const char *blockName, const char* uniformName, void *value) 
{
    if (!pBlocks.count(blockName)) return;

    std::string uniformComposed = blockName + std::string(".") + uniformName;
	std::string finalUniName;

    if (pBlocks[blockName].uniformOffsets.count(uniformName))
		finalUniName = uniformName;
    else if (pBlocks[blockName].uniformOffsets.count(uniformComposed))
		finalUniName = uniformComposed;
	else
		return;

	UniformBlock b = pBlocks[blockName];

	BlockUniform bUni = b.uniformOffsets[finalUniName];
	glBindBuffer(GL_UNIFORM_BUFFER, b.buffer);
	glBufferSubData(GL_UNIFORM_BUFFER, bUni.offset, bUni.size, value);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}


void GLProgram::setBlockUniformArrayElement(const char * blockName, 
								const char * uniformName,
								int arrayIndex, 
								void * value) 
{
	assert(pBlocks.count(blockName) && pBlocks[blockName].uniformOffsets.count(uniformName));

	UniformBlock b = pBlocks[blockName];

	BlockUniform bUni = b.uniformOffsets[uniformName];

	glBindBuffer(GL_UNIFORM_BUFFER, b.buffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 
						bUni.offset + bUni.arrayStride * arrayIndex, 
						bUni.arrayStride, value);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}



std::string GLProgram::getShaderInfoLog(ShaderType type)
{
    unsigned int shaderHdl = pShader[type];

    int length = 0;
    glGetShaderiv(shaderHdl, GL_INFO_LOG_LENGTH, &length);
    std::string s;

    if (length > 0) {
        char * c_log = new char[length];
        int nCharWritten = 0;
        glGetShaderInfoLog(shaderHdl, length, &nCharWritten, c_log);
        s = nCharWritten ? c_log : "OK";
        delete[] c_log;
    }

    return s;
}


std::string GLProgram::getProgramInfoLog()
{
    int length = 0;
    glGetProgramiv(pProgram, GL_INFO_LOG_LENGTH, &length);
    std::string s;

    if (length > 0) {
        char * c_log = new char[length];
        int nCharWritten = 0;
        glGetProgramInfoLog(pProgram, length, &nCharWritten, c_log);
        s = nCharWritten ? c_log : "OK";
        delete[] c_log;
    }

    return s;
}

std::string GLProgram::getAllInfoLog()
{	
    std::string s;

    for (int i = 0; i < NUM_SHADER_TYPE; ++i) {
		if (pShader[i]) {
            s += ShaderTypeStrs[i] + std::string(": ") + getShaderInfoLog((ShaderType)i) + "\n";
		}
	}

	if (pProgram) {
		s += "Program: " + getProgramInfoLog();
        s += validate() ? " - Valid\n" : " - Not Valid\n";
	}

	return s;
}


bool GLProgram::shaderCompiled(ShaderType type)
{
    int b = GL_FALSE;
    if (pProgram) glGetShaderiv(pShader[type], GL_COMPILE_STATUS, &b);
    return b != GL_FALSE;
}


bool GLProgram::linked()
{
	int b = GL_FALSE;
    if (pProgram) glGetProgramiv(pProgram, GL_LINK_STATUS, &b);
    return b != GL_FALSE;
}


void GLProgram::bind()
{
    assert(glIsProgram(pProgram));

    if (pProgram <= 0) return;
    glUseProgram( pProgram );
}

void GLProgram::unbind(){ glUseProgram(0); }


bool GLProgram::isbinded()
{
	int p = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &p);
    //glGetIntegerv(GL_ACTIVE_PROGRAM, &p);
    return p == pProgram;
}

int GLProgram::getAttribLocation(const char * name)
{
    assert(isbinded());

    int loc = glGetAttribLocation(pProgram, name);
    if (loc < 0) fprintf(stderr, "Attribute: %s not found.\n", name);
    return loc;
}

void GLProgram::bindAttribLocation(unsigned int location, const char * name)
{
    glBindAttribLocation(pProgram, location, name);
}

void GLProgram::bindFragDataLocation(unsigned int location, const char * name)
{
    glBindFragDataLocation(pProgram, location, name);
}

void GLProgram::printActiveUniforms() 
{
    GLint nUniforms, size, location, maxLen;
    GLsizei written;
    GLenum type;

    glGetProgramiv( pProgram, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxLen);
    glGetProgramiv( pProgram, GL_ACTIVE_UNIFORMS, &nUniforms);

    GLchar *name = new GLchar[maxLen];

    printf(" Location | Name\n");
    printf("------------------------------------------------\n");
    for( int i = 0; i < nUniforms; ++i ) {
        glGetActiveUniform( pProgram, i, maxLen, &written, &size, &type, name );
        location = glGetUniformLocation(pProgram, name);
        printf(" %-8d | %s\n",location, name);
    }

    delete []name;
}

void GLProgram::printActiveAttribs() 
{
    GLint written, size, location, maxLength, nAttribs;
    GLenum type;

    glGetProgramiv(pProgram, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &maxLength);
    glGetProgramiv(pProgram, GL_ACTIVE_ATTRIBUTES, &nAttribs);

    GLchar *name = new GLchar[maxLength];

    printf(" Index | Name\n");
    printf("------------------------------------------------\n");
    for( int i = 0; i < nAttribs; i++ ) {
        glGetActiveAttrib( pProgram, i, maxLength, &written, &size, &type, name );
        location = glGetAttribLocation(pProgram, name);
        printf(" %-5d | %s\n",location, name);
    }

    delete []name;
}

bool GLProgram::validate()
{
    if (!linked()) return false;

    glValidateProgram(pProgram);
    GLint status;
    glGetProgramiv(pProgram, GL_VALIDATE_STATUS, &status);

    if (GL_FALSE != status)  return true;
    
    // Fail to validate, store log and return false
    logString = getProgramInfoLog();
    return false;
}

