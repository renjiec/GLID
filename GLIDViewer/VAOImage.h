#pragma once

#include "MyImage.h"
#include <algorithm>


#ifdef _DEBUG
#include <cassert>
#define MAKESURE(x) assert(x)
#else
#define MAKESURE(x) 
#endif

class GLQuad
{
private:
    unsigned int vaoHandle;

public:
	GLQuad() :vaoHandle(0) {
        // make sure OpenGL Context is created before this!

		// Set up the buffers
		unsigned int handle[2];
		glGenBuffers(2, handle);

		// Set up the VAO(vertex array object)
		glGenVertexArrays(1, &vaoHandle);
		glBindVertexArray(vaoHandle);

		// Vertex position
		glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
		const GLbyte x[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(x), x, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 2, GL_BYTE, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(0);

		// Texture coordinates
		glBindBuffer(GL_ARRAY_BUFFER, handle[1]);
		const GLbyte uv[] = { 0, 0, 1, 0, 0, 1, 1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(uv), uv, GL_STATIC_DRAW);
		glVertexAttribPointer(2, 2, GL_BYTE, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(2);

		glBindVertexArray(0);
	}

	~GLQuad() { glDeleteVertexArrays(1, &vaoHandle); }

	void draw() const {
		glBindVertexArray(vaoHandle);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // GL_QUADS not supported by glDrawArrays
		//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBindVertexArray(0);
	}

	static const char* vertexShaderStr() {
        return R"(#version 330
                layout (location = 0) in vec2 VertexPosition;
                layout (location = 2) in vec2 VertexTexCoord;
                uniform mat3 modelview;
                out vec2 TexCoord;
                void main(){
                    gl_Position = vec4(modelview*vec3(VertexPosition,1), 1);
                    TexCoord = VertexTexCoord;
                })";
	}

	static const char* fragShaderStr() {
        return R"(#version 330
                in vec2 TexCoord;
                out vec4 FragColor;
                uniform sampler2D img;
                uniform float alpha=1.0;
                void main(){ FragColor = vec4(texture(img, TexCoord).rgb, alpha); })";
	}

};

inline void drawUnitQuad()
{
	static GLQuad quad;
	quad.draw();
}




int imDim2glTexFormat(int d) 
{
    //const int formats[][4] = { { GL_RED, GL_RG, GL_RGB, GL_RGBA }, { GL_COMPRESSED_RGB, GL_COMPRESSED_RG, GL_COMPRESSED_RGB, GL_COMPRESSED_RGBA } };
    const int formats[] = { GL_RED, GL_RG, GL_RGB, GL_RGBA };
    return formats[std::max(d - 1, 0)];
}

int imDim2glTexInputFormat(int d) {
    const int formats[] = { GL_RED, GL_RG, GL_RGB, GL_RGBA };
    return formats[std::max(d - 1, 0)];
}



class GLTexture
{
public:
	unsigned int handle;
    unsigned int nMultisample;

    GLTexture(const MyImage* img = nullptr, int ms = 1) : handle(0),nMultisample(ms)	{ if (img) setImage(*img); }
    GLTexture(GLTexture&& t)    { swap(std::move(t)); }

    GLTexture(const GLTexture&) = delete;
    GLTexture& operator =(const GLTexture&) = delete;

	~GLTexture() { glDeleteTextures(1, &handle); }

    void swap(GLTexture&& t)    { std::swap(handle, t.handle); }

    GLenum target() const { return nMultisample>1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D; }

    static GLTexture fromFile(const std::string &filename)    { MyImage tmp(filename); return GLTexture(&tmp); }

    static GLTexture genWhiteTexture(int w=16, int h=16) 
    {
        //int align;
        //glGetIntegerv(GL_UNPACK_ALIGNMENT, &align);
        //int pitch = (w * 3 + align - 1) / align*align;

        ////Upload pixel data.
        //std::vector<unsigned char> pixels(h * pitch, 255);
        //return GLTexture(&MyImage(pixels.data(), w, h, pitch, 3));

        GLTexture tex;
        tex.allocateStorage(w, h, GL_RGB, GL_RGB);
        unsigned char col[] = { 255, 255, 255, 255 };
        tex.clearToColor(col);
        return tex;
    }

    void clearToColor(const unsigned char *col)    {
#if OPENGL_VERSION >= 4
        glClearTexImage(handle, 0, GL_RGB, GL_UNSIGNED_BYTE, col); // only available on OpenGL4 hardware
#else
        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, handle, 0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        GLuint clearcolor[] = { col[0], col[1], col[2], col[3] };
        glClearBufferuiv(GL_COLOR, 0, clearcolor);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
#endif
    }

	void allocateStorage(int w, int h, int internalFormat, int format=GL_RGB, int filtering = GL_LINEAR, int clamping = GL_REPEAT);
    void setImage(const MyImage &img, bool allocated=false, int internalFormat=0);

	void bind()     { glBindTexture(target(), handle); }
	void unbind()   { glBindTexture(target(), 0); }
    bool isbinded() const { int v;  glGetIntegerv(GL_TEXTURE_BINDING_2D, &v); return v == handle; }
    int  width() const 	{ assert(isbinded()); int i; glGetTexLevelParameteriv(target(), 0, GL_TEXTURE_WIDTH, &i); return i; }
	int  height() const	{ assert(isbinded()); int i; glGetTexLevelParameteriv(target(), 0, GL_TEXTURE_HEIGHT, &i); return i; }
    int  format() const { assert(isbinded()); int i; glGetTexLevelParameteriv(target(), 0, GL_TEXTURE_INTERNAL_FORMAT, &i); return i; }
    operator MyImage();

	int filtering() {

		bind();
		int v, v2;
		glGetTexParameteriv(target(), GL_TEXTURE_MIN_FILTER, &v);
		glGetTexParameteriv(target(), GL_TEXTURE_MAG_FILTER, &v2);
		MAKESURE(v == v2);
		return v;  
	}

	int clamping() {
		bind();
		int v, v2;
		glGetTexParameteriv(target(), GL_TEXTURE_WRAP_S, &v);
		glGetTexParameteriv(target(), GL_TEXTURE_WRAP_T, &v2);
		MAKESURE(v == v2);
		return v;
	}

	void setFiltering(int v) 
	{
		bind();
		glTexParameteri(target(), GL_TEXTURE_MIN_FILTER, v);
		glTexParameteri(target(), GL_TEXTURE_MAG_FILTER, v);
	}

	void setClamping(int v) 
	{
		bind();
		glTexParameteri(target(), GL_TEXTURE_WRAP_S, v);
		glTexParameteri(target(), GL_TEXTURE_WRAP_T, v);
	}

    void setFormat(unsigned int v)
    {
        bind();
        if (handle && v != format()){ allocateStorage(width(), height(), v); }
    }
};

GLTexture::operator MyImage()
{
    bind();
    const int w = width();
    const int h = height();

    //download pixel data.
    int align;
    glGetIntegerv(GL_PACK_ALIGNMENT, &align);

    //int pitch = (w * 3 + align - 1) / align*align;
    int pitch = (w * 4 + align - 1) / align*align;
    std::vector<BYTE> pixels(h*pitch);
    //glGetTexImage(target(), 0, GL_BGR, GL_UNSIGNED_BYTE, &pixels[0]);
    glGetTexImage(target(), 0, GL_RGBA, GL_UNSIGNED_BYTE, &pixels[0]);

    return MyImage(pixels.data(), w, h, pitch, 4);
}


void GLTexture::allocateStorage(int w, int h, int internalFormat, int inputFormat, int filtering, int clamping)
{
	if (handle)
		glDeleteTextures(1, &handle); // clean previous memory

	glGenTextures(1, &handle);
	glBindTexture(target(), handle);

	glTexParameteri(target(), GL_TEXTURE_MIN_FILTER, filtering);
	glTexParameteri(target(), GL_TEXTURE_MAG_FILTER, filtering);
	glTexParameteri(target(), GL_TEXTURE_WRAP_S, clamping);
	glTexParameteri(target(), GL_TEXTURE_WRAP_T, clamping);

    if (internalFormat == GL_RED){  // for grayscale images
        GLint cols[] = { GL_RED, GL_RED, GL_RED, 1 };
        glTexParameteriv(target(), GL_TEXTURE_SWIZZLE_RGBA, cols);
    }
    // Allocate the storage. use RGB (no alpha channel)
    if (nMultisample>1)
        glTexImage2DMultisample(target(), nMultisample, GL_RGBA8, w, h, false);
    else
        glTexImage2D(target(), 0, internalFormat, w, h, 0, inputFormat, GL_UNSIGNED_BYTE, nullptr);
}

void GLTexture::setImage(const MyImage &img, bool allocated, int internalFormat)
{
    bind();
    if (!internalFormat) internalFormat = imDim2glTexFormat(img.dim());
    if (!allocated)  allocateStorage(img.width(), img.height(), internalFormat);
    //allocateStorage(img.width(), img.height(), internalFormat, dim2inputFormat(img.dim()));

    const int w = img.width();
    const int h = img.height();
    MAKESURE(w == width() && h == height());
 
    glPixelStorei(GL_UNPACK_ALIGNMENT, MyImage::alignment()); // for texture uploading
    
    int format = imDim2glTexInputFormat(img.dim());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, format, GL_UNSIGNED_BYTE, img.data());

    const GLfloat white[] = { 1.f, 1.f, 1.f, 1.f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, white);
}


class GLArrayTexture
{
public:
	unsigned int handle;
	//unsigned int format;        // GL_RGB8, GL_RGB10, GL_RGB12, GL_RGB16, GL_RGB16F, GL_RGB32F, GL_RGB32, GL_COMPRESSED_RGB
	//unsigned int clamping;      // GL_CLAMP_TO_EDGE, GL_CLAMP_TO_BORDER, GL_REPEAT, GL_MIRRORED_REPEAT
	//unsigned int filtering;     // GL_LINEAR, GL_NEAREST

    GLArrayTexture(const std::vector<MyImage> *img = nullptr) : handle(0)	{ if (img) setImage(*img); }
    GLArrayTexture(const GLArrayTexture&) = delete;
    GLArrayTexture& operator =(const GLArrayTexture&) = delete;

	~GLArrayTexture() { glDeleteTextures(1, &handle); }

	void allocateStorage(int w, int h, int nlayer, int format, int inputFormat=GL_RGB, int filtering = GL_LINEAR, int clamping = GL_REPEAT);
        operator std::vector<MyImage>();
	void clearToColor(const unsigned char *col, int iLayer=-1);
    int  loadFromFiles(const std::vector<std::string> &filenames, int internalFormat = 0);
	void setImage(const std::vector<MyImage> &imgs, bool allocated=false, int internalFormat = 0);
	void bind()     { MAKESURE(glIsTexture(handle));  glBindTexture(GL_TEXTURE_2D_ARRAY, handle); }
	void unbind()   { glBindTexture(GL_TEXTURE_2D_ARRAY, 0); }
	int  width() 	{ bind();  int i; glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH,  &i); return i; }
	int  height()	{ bind();  int i; glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT, &i); return i; }
	int  numLayers(){ bind();  int i; glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH,  &i); return i; }

	int filtering() {
		bind();
		int v, v2;
		glGetTexParameteriv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, &v);
		glGetTexParameteriv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, &v2);
		MAKESURE(v == v2);
		return v;  
	}

	int clamping() {
		bind();
		int v, v2;
		glGetTexParameteriv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, &v);
		glGetTexParameteriv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, &v2);
		MAKESURE(v == v2);
		return v;
	}

    int format() { bind();  int i; glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_INTERNAL_FORMAT, &i); return i; }

	void setFiltering(int v) 
	{
		bind();
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, v);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, v);
	}

	void setClamping(int v) 
	{
		bind();
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, v);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, v);
	}

    void setFormat(unsigned int v)
    {
        if (handle && v != format()){ bind(); allocateStorage(width(), height(), numLayers(), v); }
    }
};


void GLArrayTexture::allocateStorage(int w, int h, int nlayer, int internalFormat, int format, int filtering, int clamping)
{
	if (handle)
		glDeleteTextures(1, &handle); // clean previous memory

	glGenTextures(1, &handle);
	glBindTexture(GL_TEXTURE_2D_ARRAY, handle);

	//Allocate the storage. use RGB (no alpha channel)
	//glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, format, w, h, nlayer); // does not handle compression?
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, internalFormat, w, h, nlayer, 0, format, GL_UNSIGNED_BYTE, nullptr);

	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, filtering);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, filtering);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, clamping);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, clamping);

    if (internalFormat == GL_RED){  // for grayscale images
        GLint cols[] = { GL_RED, GL_RED, GL_RED, 1 };
        glTexParameteriv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_SWIZZLE_RGBA, cols);
    }
}

void GLArrayTexture::clearToColor(const unsigned char *col, int iLayer)
{
    const int w = width();
    const int h = height();

#if OPENGL_VERSION >= 4
    if (iLayer < 0) 
        glClearTexImage(handle, 0, GL_RGB, GL_UNSIGNED_BYTE, col);
    else
        glClearTexSubImage(handle, 0, 0, 0, iLayer, w, h, 1, GL_RGB, GL_UNSIGNED_BYTE, col);
#else
    int nLayer = numLayers();
    //int align;
    //glGetIntegerv(GL_UNPACK_ALIGNMENT, &align);
    //int pitch = (w * 3 + align - 1) / align*align;
    //std::vector<unsigned char> pixels(h * pitch * nLayer);
    //for (int i = 0; i < h*nLayer; i++) {
    //	for (int j = 0; j < w; j++)
    //       std::copy_n(col, 3, &pixels[i*pitch + j * 3]);
    //}
    //
    ////Upload pixel data.
    //glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, w, h, nLayer, GL_RGB, GL_UNSIGNED_BYTE, &pixels[0]);

    // TODO: not tested working
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

    GLuint clearcolor[] = { col[0], col[1], col[2], col[3] };
    for (int i = 0; i < nLayer; i++){
        if (iLayer >= 0 && iLayer != i) continue;
        glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, handle, 0, i);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glClearBufferuiv(GL_COLOR, 0, clearcolor);
    }
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
#endif
}
	
GLArrayTexture::operator std::vector<MyImage>()
{
	bind();
    const int w = width();
    const int h = height();
    const int nLayers = numLayers();

	//download pixel data.
	int align;
	glGetIntegerv(GL_PACK_ALIGNMENT, &align);
	int pitch = (w * 3 + align - 1) / align*align;

	std::vector<BYTE> pixels(h*pitch*nLayers);
	glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, GL_UNSIGNED_BYTE, &pixels[0]);
	
	std::vector<MyImage> imgs(nLayers);
	for (int i = 0; i < nLayers; i++)
		imgs[i] = MyImage(&pixels[i*pitch*h], w, h, pitch);

	return imgs;
}


void GLArrayTexture::setImage(const std::vector<MyImage> &imgs, bool allocated, int internalFormat)
{
    if (imgs.empty()) return;

    if (!internalFormat) internalFormat = imDim2glTexFormat(imgs[0].dim());
    if (!allocated)	allocateStorage(imgs[0].width(), imgs[0].height(), int(imgs.size()), internalFormat);

	const int nLayer = int(imgs.size());
    int w = imgs[0].width();
    int h = imgs[0].height();

	MAKESURE(w==width() && h==height() && nLayer<=numLayers()); // fixed, may have more layered allocated in the beginning
 
	//Upload pixel data.
	glPixelStorei(GL_UNPACK_ALIGNMENT, MyImage::alignment()); // for texture uploading

    int format = imDim2glTexInputFormat(imgs[0].dim());
	for (int iLayer = 0; iLayer < nLayer; iLayer++){
		glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, iLayer, w, h, 1, format, GL_UNSIGNED_BYTE, imgs[iLayer].data());
	}

}


int GLArrayTexture::loadFromFiles(const std::vector<std::string> &filenames, int internalFormat)
{
	if (filenames.empty()) return 0;

	const int nLayer = int(filenames.size());

    MyImage img(filenames[0]);
    int w = img.width();
    int h = img.height();

    if (!internalFormat) internalFormat = imDim2glTexFormat(img.dim());
    allocateStorage(w, h, nLayer, internalFormat);
	MAKESURE(w==width() && h==height() && nLayer==numLayers());
 
	//Upload pixel data.
    if (MyImage::alignment() > 8) 
        fprintf(stderr, "!MyImage has too big alignment(%d)!\n", MyImage::alignment());

	glPixelStorei(GL_UNPACK_ALIGNMENT, MyImage::alignment()); // for texture uploading

    printf("Loading %d textures from disk...", nLayer);
	for (int iLayer = 0; iLayer < nLayer; iLayer++){
        if (iLayer>0) img = MyImage(filenames[iLayer]);
		glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, iLayer, w, h, 1, GL_RGB, GL_UNSIGNED_BYTE, img.data());
	}

    printf("finished!\n");

    return nLayer;
}

class GLFboMs
{
public:
    GLuint tex, fbo;

    GLFboMs(int w, int h, int nSample)
    {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);

        // Allocate the storage. use RGB (no alpha channel)
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, nSample, GL_RGBA8, w, h, false);

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        //glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
        //glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        //glFramebufferRenderbuffer()
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tex, 0);
    }

    ~GLFboMs(){
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
    }
};


class GLTextureBuffer
{
public:
	unsigned int texHandle;
	unsigned int bufHandle;

    GLTextureBuffer() : texHandle(0), bufHandle(0){ }
    GLTextureBuffer(GLTextureBuffer&& t)    { swap(std::move(t)); }

    GLTextureBuffer(const GLTextureBuffer&) = delete;
    GLTextureBuffer& operator =(const GLTextureBuffer&) = delete;

	~GLTextureBuffer()
	{
	    glDeleteTextures(1, &texHandle);
	    glDeleteBuffers(1, &bufHandle);
	}

    void swap(GLTextureBuffer&& t)
	{
	    std::swap(texHandle, t.texHandle);
        std::swap(bufHandle, t.bufHandle);
	}

	void bind()     { glBindTexture(GL_TEXTURE_BUFFER, texHandle); }
	void unbind()   { glBindTexture(GL_TEXTURE_BUFFER, 0); }

    void upload(const void *data, int datasize, int datatype)
    {
        if (!bufHandle) glGenBuffers(1, &bufHandle);
        glBindBuffer(GL_TEXTURE_BUFFER, bufHandle);
        glBufferData(GL_TEXTURE_BUFFER, datasize, NULL, GL_STATIC_DRAW);
        glBufferSubData(GL_TEXTURE_BUFFER, 0 , datasize, data);

        if (!texHandle) glGenTextures(1, &texHandle);
        bind();
        glTexBuffer(GL_TEXTURE_BUFFER, datatype, bufHandle);
    }

};