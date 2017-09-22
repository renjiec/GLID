#pragma once
#include <algorithm>
#include <vector>
#include <array>

//#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <Eigen/LU>
#include "glprogram.h"
#include "glarray.h"


const unsigned char jetmaprgb[] = {
	143, 0, 0,
159, 0, 0,
175, 0, 0,
191, 0, 0,
207, 0, 0,
223, 0, 0,
239, 0, 0,
255, 0, 0,
255, 15, 0,
255, 31, 0,
255, 47, 0,
255, 63, 0,
255, 79, 0,
255, 95, 0,
255, 111, 0,
255, 127, 0,
255, 143, 0,
255, 159, 0,
255, 175, 0,
255, 191, 0,
255, 207, 0,
255, 223, 0,
255, 239, 0,
255, 255, 0,
239, 255, 15,
223, 255, 31,
207, 255, 47,
191, 255, 63,
175, 255, 79,
159, 255, 95,
143, 255, 111,
127, 255, 127,
111, 255, 143,
95, 255, 159,
79, 255, 175,
63, 255, 191,
47, 255, 207,
31, 255, 223,
15, 255, 239,
0, 255, 255,
0, 239, 255,
0, 223, 255,
0, 207, 255,
0, 191, 255,
0, 175, 255,
0, 159, 255,
0, 143, 255,
0, 127, 255,
0, 111, 255,
0, 95, 255,
0, 79, 255,
0, 63, 255,
0, 47, 255,
0, 31, 255,
0, 15, 255,
0, 0, 255,
0, 0, 239,
0, 0, 223,
0, 0, 207,
0, 0, 191,
0, 0, 175,
0, 0, 159,
0, 0, 143,
0, 0, 127 };


template<typename R=float, int dimension=2>
struct GLMesh
{
    enum{ dim = dimension };
    using MapMat4 = Eigen::Map < Eigen::Matrix < float, 4, 4, Eigen::RowMajor > > ;
    using ConstMat4 = Eigen::Map < const Eigen::Matrix < float, 4, 4, Eigen::RowMajor > > ;

    enum PickingElements {PE_NONE=0, PE_VERTEX, PE_FACE};
    enum PickingOperations {PO_NONE=0, PO_ADD, PO_REMOVE};

    struct Mesh
    {
        static const int dim = 2;
        std::vector<R> X, UV;
        std::vector<int> T;
        size_t nVertex() const { return X.size() / 2; }
        size_t nFace() const { return T.size() / 3; }
    };
    Mesh mesh;
    GLTexture tex;
    static GLTexture colormapTex;

    typedef std::array<R, 4> vec4;
    //typedef std::array<R, 3> vec3;
    typedef std::array<R, dim> vec;

    int nVertex;
    int nFace;
    GLuint vaoHandle;

    GLArray<R, dim> gpuX;
    GLArray<int, 3, true> gpuT;
    GLArray<R, 2>   gpuUV;
    GLArray<R, dim> gpuBaryCenters;
    GLArray<R, 1>   gpuVertexData;
    GLArray<R, 1>   gpuFaceData;
    R vtxDataMinMax[2];
    bool vizVtxData;

    std::map<int, vec> constrainVertices, constrainVerticesRef;
    std::vector<R> vertexData;
    std::vector<R> faceData;
    std::vector<R> vertexVF;

    int actVertex;
    int actFace;
    int actConstrainVertex;

    vec4 faceColor;
    vec4 edgeColor;
    vec4 vertexColor;
    int depthMode = 0;
    float edgeWidth;
    float pointSize;

    float auxPointSize;
    float vertexDataDrawScale;
    float faceDataDrawScale;
    float VFDrawScale;

    float mMeshScale;
    float mTextureScale;

    std::vector<int> auxVtxIdxs;

    vec mTranslate;
    R bbox[dim * 2];

	bool showTexture;
    bool drawTargetP2P = true;

    static GLProgram prog, pickProg, pointSetProg;

    ~GLMesh() { glDeleteVertexArrays(1, &vaoHandle); }

    GLMesh() :vaoHandle(0), showTexture(false), mMeshScale(1.f), mTranslate({ 0.f }), mTextureScale(1.f), edgeWidth(1), pointSize(0), actVertex(-1), actFace(-1), 
        actConstrainVertex(-1), faceColor({ 1.f, 1.f, 1.f, 1.f }), edgeColor({ 0.f, 0.f, 0.f, 0.1f }), vertexColor({ 1.f, 0.f, 0.f, .1f }), vertexDataDrawScale(0.f), 
        faceDataDrawScale(0.f), VFDrawScale(0.f), vizVtxData(0), auxPointSize(0.f)
    {}

	void allocateStorage(int nv, int nf)
	{
		if (nv == nVertex && nf == nFace) return;
        nVertex = nv;
        nFace = nf;
        //mesh.X.resize(nv * dim);
        //mesh.UV.resize(nv * 2);
        //mesh.T.resize(nf * 3);

		if (!vaoHandle)  glGenVertexArrays(1, &vaoHandle);

        glBindVertexArray(vaoHandle);

        gpuX.allocateStorage(nv);
        glVertexAttribPointer(0, dim, GLType<R>::val, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);  // Vertex position

        gpuUV.allocateStorage(nv);
		glVertexAttribPointer(2, 2, GLType<R>::val, GL_FALSE, 0, nullptr);
		glEnableVertexAttribArray(2);  // Texture coords

        gpuVertexData.allocateStorage(nv);
		glVertexAttribPointer(4, 1, GLType<R>::val, GL_FALSE, 0, nullptr);
		glEnableVertexAttribArray(4);  // Vertex data

        gpuT.allocateStorage(nf);

        glBindVertexArray(0);

        // barycenter coordinates
        gpuBaryCenters.allocateStorage(nf);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
		glEnableVertexAttribArray(0);
	}

    std::vector<int> getConstrainVertexIds() const{
        std::vector<int> idxs;
        idxs.reserve(constrainVertices.size());
        for (auto it : constrainVertices)   idxs.push_back(it.first);
        return idxs;
    }

    std::vector<R> getConstrainVertexCoords() const{
        //std::vector<R> x(nVertex*dim);
        //gpuX.downloadData(x.data(), nVertex);
        std::vector<R> x;
        x.reserve(constrainVertices.size()*dim);
        for (auto it : constrainVertices) {
            x.push_back(it.second[0]);
            x.push_back(it.second[1]);
        }
        return x;
    }

    std::vector<R> genVertexVF(const std::vector<R> &vfd, float scale) {
        const std::vector<R> x = mesh.X;

        std::vector<R> vf;
        vf.reserve(nVertex*dim * 2);
        for (int i = 0; i < nVertex; i++){
            vf.insert(vf.end(), { x[i*dim], x[i*dim + 1], x[i*dim] + vfd[i*dim]*scale, x[i*dim + 1]+vfd[i*dim+1]*scale });
        }

        return vf;
    }


    void getTriangulation(int *t)    { gpuT.downloadData(t, nFace); }

    void getTriangulation(int ti, int *t)    { gpuT.at(ti, t); }

    void getVertex(R *x)  { gpuX.downloadData(x, nVertex); }

    void getVertex(int vi, R *x)   { gpuX.at(vi, x); }

    void setVertex(int vi, const R *x)
    {
        //for (int i = 0; i < dim; i++)    mesh.X[vi*dim + i] = x[i];
        gpuX.setAt(vi, x);
    }

    void setConstraintVertices(const int *ids, const R* pos, size_t nc){
        constrainVertices.clear();
        for (size_t i = 0; i < nc; i++) constrainVertices.insert({ ids[i], vec{ pos[i * 2], pos[i * 2 + 1] } });
    }

    void setVertexDataViz(const R* val)
    {
        vizVtxData = (nullptr != val);
        prog.bind();
        prog.setUniform("colorCoding", int(vizVtxData));

        if (vizVtxData) {
            gpuVertexData.uploadData(val, mesh.nVertex());

            glBindVertexArray(vaoHandle);
            gpuVertexData.bind();
            glVertexAttribPointer(5, 1, GLType<R>::val, GL_FALSE, 0, nullptr);
            glEnableVertexAttribArray(5);

            auto mm = std::minmax_element(val, val + mesh.nVertex());
            vtxDataMinMax[0] = *mm.first;
            vtxDataMinMax[1] = *mm.second;
            prog.setUniform("dataMinMax", vtxDataMinMax[0], vtxDataMinMax[1]);
        }
        else{
            glBindVertexArray(vaoHandle);
            glDisableVertexAttribArray(5);
        }
    }

    void updateDataVizMinMax()
    {
        prog.bind();
        prog.setUniform("dataMinMax", vtxDataMinMax[0], vtxDataMinMax[1]);
    }

    template<class MatrixR, class MatrixI>
	void upload(const MatrixR &x, const MatrixI &t, const R *uv)
    {
        upload(x.data(), (int)x.rows(), t.count()?t.data():nullptr, (int)t.rows(), uv);
    }

    void upload(const R *x, int nv, const int *t, int nf, const R *uv)
    {
        allocateStorage(x ? nv : nVertex, t ? nf : nFace);
        //if (x  && x!=mesh.X.data()) std::copy_n(x, nv*dim, mesh.X.data());
        //if (uv && uv!=mesh.UV.data()) std::copy_n(uv, nv*2, mesh.UV.data());
        //if (t  && t!=mesh.T.data()) std::copy_n(t, nf*3, mesh.T.data());

        if (x){ gpuX.uploadData(x, nv, false); mesh.X.assign(x, x + nv*dim); }
        if (uv){ gpuUV.uploadData(uv, nv, false); mesh.UV.assign(uv, uv + nv * 2);}
        if (t){ gpuT.uploadData(t, nf, false); mesh.T.assign(t, t + nf * 3); }

        if (x&&t){
            boundingbox(x, nv, bbox);  // update bounding box for the initialization
            constrainVertices.clear();
            constrainVerticesRef.clear();
            auxVtxIdxs.clear();

            actVertex = -1;
            actFace = -1;
            actConstrainVertex = -1;

            vertexData = std::vector<R>(nv, 0);
            faceData = std::vector<R>(nf, 1);

            gpuVertexData.uploadData(vertexData.data(), nVertex, false);
            gpuFaceData.uploadData(faceData.data(), nFace);
        }

        if (x)
            gpuBaryCenters.uploadData(baryCenters(x).data(), nFace);
	}

    void updateBBox(){
        boundingbox(mesh.X.data(), nVertex, bbox);
    }

    std::vector<R> baryCenters(const R* X)
    {
        std::vector<R> x(nFace * dim);
        for (int i = 0; i < nFace; i++){
            const R *px[] = { &X[mesh.T[i * 3] * dim], &X[mesh.T[i * 3 + 1] * dim], &X[mesh.T[i * 3 + 2] * dim] };
            for (int j = 0; j < dim; j++) x[i*dim+j] = (px[0][j] + px[1][j] + px[2][j])/3;
        }

        return x;
    }


    float actVertexData() const { return (actVertex >= 0 && actVertex < nVertex) ? vertexData[actVertex] : std::numeric_limits<float>::infinity(); }
    float actFaceData() const { return (actFace >= 0 && actFace < nFace) ? faceData[actFace] : std::numeric_limits<float>::infinity(); }
    void incActVertexData(float pct){ setVertexData(actVertex, (vertexData[actVertex]+1e-3f) * (1 + pct)); }
    void incActFaceData(float pct){ setFaceData(actFace, (faceData[actFace]+1e-3f) * (1 + pct)); }

    void setVertexData(int i, R v){
        MAKESURE(i < nVertex && i >= 0);
        gpuVertexData.setAt(i, &v);
        vertexData[i] = v;
    }

    void setFaceData(int i, R v){
        MAKESURE(i < nFace && i >= 0);
        gpuFaceData.setAt(i, &v);
        faceData[i] = v;
    }

    void setVertexData(const R *vtxData)    { gpuVertexData.uploadData(vtxData, nVertex, false); }
    void setFaceData(const R *vtxData)    { gpuFaceData.uploadData(vtxData, nFace, false); }

    bool showWireframe() const {  return edgeWidth > 0; }
    bool showVertices() const {  return pointSize > 0; }

    R drawscale() const
    {
        R scale0 = 1.9f / std::max(bbox[dim] - bbox[0], bbox[1 + dim] - bbox[1]);
        return mMeshScale*scale0; 
    }

    double pixelDistance(const int *vp) const
    {
        ConstMat4 mv(modelViewMat(vp).data());
        return std::max(1. / mv(0, 0) / vp[2], 1. / mv(1, 1) / vp[3]);
    }

    std::array<R, 2> mapWin2World(int x, int y, const int *vp) const
    {
        const R p[] = { x*2.f / vp[2] - 1, 1 - y*2.f / vp[3] };
        using Mat = Eigen::Map < const Eigen::Matrix<R,2,4,Eigen::RowMajor> > ;
        Eigen::Vector2f p0 = Mat(invModelView(vp).data())*(Eigen::Vector4f() << p[0], p[1], 0, 1).finished();
        return std::array < R, 2 > {p0[0], p0[1]};
    }

    std::array<R, 16> invModelView(const int *vp) const{
        auto mv = modelViewMat(vp);
        mv[3] /= -mv[0];
        mv[7] /= -mv[5];
        mv[0] = 1 / mv[0];
        mv[5] = 1 / mv[5];
        return mv;
    }

    std::array<R, 16> modelViewMat(const int *vp, bool colmajor=false) const{
        R vscaling = vp[2] / R(vp[3]);
        R trans[] = { (bbox[0] + bbox[dim]) / 2, (bbox[1] + bbox[1+dim]) / 2 };
        R xform[2] = { 1, 1 };
        if (vscaling < 1)
            xform[1] = vscaling;
        else
            xform[0] = 1 / vscaling;

        R ss = drawscale();  //todo figureout the correct scaling which adapt to aspect ratio of the window
        R s[2] = {ss*xform[0], ss*xform[1]};
        R t[] = { mTranslate[0] - trans[0], mTranslate[1] - trans[1] };
        return colmajor?std::array<R, 16>{
            s[0], 0, 0, 0, 
            0, s[1], 0, 0,
            0, 0, 1, 0,
            t[0] * s[0], t[1] * s[1], 0, 1 }
            :std::array<R, 16>{
            s[0], 0, 0, t[0] * s[0],
            0, s[1], 0, t[1] * s[1],
            0, 0, 1, 0,
            0, 0, 0, 1 };
    }

    void draw(const int *vp)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendFunc(GL_ONE, GL_ONE);

        glActiveTexture(GL_TEXTURE0);
        tex.bind();

        prog.bind();
        prog.setUniform("depthMode", depthMode);
        prog.setUniform("textureScale", mTextureScale);	
        prog.setUniform("modelview", modelViewMat(vp).data());
        prog.setUniform("useTexMap", int(showTexture));
        prog.setUniform("color", faceColor.data());

        glBindVertexArray(vaoHandle);

        glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, nullptr); // not GL_INT

        glDisable(GL_DEPTH_TEST);

        if (showWireframe()){
            glLineWidth(edgeWidth);
            prog.setUniform("color", edgeColor.data());
            prog.setUniform("useTexMap", 0);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, nullptr); // not GL_INT
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        if (showVertices()){
            glPointSize(pointSize*mMeshScale);
            prog.setUniform("useTexMap", 0);
            prog.setUniform("color", vertexColor.data());
            glDrawArrays(GL_POINTS, 0, nVertex);
        }

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        prog.bind();
        if (!constrainVertices.empty()){
            //glBindVertexArray(vaoHandle);
            //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            prog.setUniform("useTexMap", 0);
           //prog.setUniform("vertexDataScale", 1);

            glDisable(GL_PROGRAM_POINT_SIZE);
            glPointSize( pointSize + 12 );
 
            const auto idxs = getConstrainVertexIds();
            GLArray<R, dim> consX(getConstrainVertexCoords().data(), idxs.size());

            glVertexAttribPointer(0, dim, GLType<R>::val, GL_FALSE, 0, nullptr);
            glEnableVertexAttribArray(0);  // Vertex position
            prog.setUniform("color", 0.f, 1.f, 1.f, .8f);


            if(drawTargetP2P)
                glDrawArrays(GL_POINTS, 0, (GLsizei)idxs.size());

			gpuX.bind();
			glVertexAttribPointer(0, dim, GLType<R>::val, GL_FALSE, 0, nullptr);
			glEnableVertexAttribArray(0);  // Vertex position

            //gpuT.bind();
            //glBindVertexArray(0);

            // make sure this is before the next draw
            if (actConstrainVertex >= 0){
                prog.setUniform("color", 1.f, 0.f, 1.f, .8f);
                const int id = idxs[actConstrainVertex];
                glDrawElements(GL_POINTS, 1, GL_UNSIGNED_INT, &id);
            }

            glPointSize( pointSize + 5 );
            prog.setUniform("color", 0.f, 0.f, 0.f, .8f);
            glDrawElements(GL_POINTS, (GLsizei)idxs.size(), GL_UNSIGNED_INT, idxs.data());

            glPointSize(1.f);
        }


        if (!constrainVerticesRef.empty()){
            std::vector<R> points;
            int nc = constrainVerticesRef.size();
            points.reserve(nc * 4);
            for (auto it : constrainVerticesRef){
                points.insert(points.end(), { mesh.X[it.first*dim], mesh.X[it.first*dim + 1],
                    it.second[0], it.second[1] });
            }

            GLArray<R, dim> gpuPts(points.data(), nc*2);
            glVertexAttribPointer(0, dim, GLType<R>::val, GL_FALSE, 0, nullptr);
            glEnableVertexAttribArray(0);  // Vertex position

            prog.setUniform("color", 0.f, 0.f, 1.f, .8f);
            prog.setUniform("useTexMap", 0);

            glDrawArrays(GL_LINES, 0, (GLsizei)nc*2);
        }


        prog.unbind();
    }

    int moveCurrentVertex(int x, int y, const int* vp)
    {
        if (actVertex < 0 || actVertex>=nVertex) return -1;

        auto mv = modelViewMat(vp);
        Eigen::Vector2f x1 = ConstMat4(mv.data()).inverse().eval().topRows(2)*Eigen::Vector4f(x / R(vp[2]) * 2 - 1, 1 - y / R(vp[3]) * 2, 0, 1); // Make Sure call eval before topRows

        //R v0[] = { mesh.X[pickVertex*dim], mesh.X[pickVertex*dim + 1], 0, 1 };
        //auto v1 = ConstMat4(mv.data()).transpose()*Eigen::Map<Eigen::Vector4f>(v0);

        //mesh.X[pickVertex*dim] = x1[0];
        //mesh.X[pickVertex*dim+1] = x1[1];
        //Eigen::Vector4f x2 = Eigen::Map<Eigen::Matrix4f>(mv.data()).transpose()*x1;

        //float v0[dim];
        //getVertex(pickVertex, v0);
        //setVertex(actVertex, x1.data());
        //upload(mesh.X.data(), mesh.nVertex(), nullptr, mesh.nFace(), nullptr);


        if (constrainVertices.find(actVertex) != constrainVertices.end()){
            auto it = constrainVerticesRef.find(actVertex);
            if (it != constrainVerticesRef.end()){
                Eigen::Map < Eigen::Vector2f > p(it->second.data());
                if ((x1 - p).norm() < pixelDistance(vp) * 15)
                    x1 = p;
            }

            constrainVertices[actVertex] = vec({ x1[0], x1[1] });
        }

        return 0;
    }


    std::pair<typename std::map<int, vec>::const_iterator, double> pickConstraintPoint(const R* p0, const std::map<int, vec> &p2p)
    {
        auto p2pIt = p2p.cend();
        double dist = 1e30;

        using Vec = Eigen::Map < const Eigen::Vector2f > ;
        R v[dim];
        for (auto it = p2p.cbegin(); it != p2p.cend(); ++it){
            getVertex(it->first, v);
            double d = std::min((Vec(v) - Vec(p0)).norm(), (Vec(it->second.data())-Vec(p0)).norm());
            if (d < dist){
                dist = d;
                p2pIt = it;
            }
        }

        return std::make_pair(p2pIt, dist);
    }

    //std::tuple<int, int, double> pickConstraintPointa(const R* p0)
    //{
    //    const auto xi = getConstrainVertexIds();
    //    size_t n = xi.size();

    //    const auto x = getConstrainVertexCoords();

    //    using Vec = Eigen::Map < const Eigen::Vector2f > ;
    //    std::vector<double> dist(n);
    //    R v[dim];
    //    for (size_t i = 0; i < n; i++){
    //        getVertex(xi[i], v);
    //        dist[i] = std::min((Vec(v) - Vec(p0)).norm(), (Vec(&x[i*dim])-Vec(p0)).norm());
    //    }
    //    int i = int( std::min_element(dist.cbegin(), dist.cend()) - dist.cbegin() );
    //    return std::make_tuple(i, xi[i], dist[i]);
    //}



    int pick(int x, int y, const int *vp, int pickElem, int operation)
    {
        int idx = -1;
        if (pickElem == PE_VERTEX){
            auto pickres = pickConstraintPoint(mapWin2World(x, y, vp).data(), constrainVertices);

            double pickThresh = pixelDistance(vp) * 20;
            if (pickres.second < pickThresh && pickres.first != constrainVertices.cend()) {
                idx = pickres.first->first;
            }
            else if (!constrainVerticesRef.empty()){
                pickres = pickConstraintPoint(mapWin2World(x, y, vp).data(), constrainVerticesRef);
                if (pickres.second < pickThresh && pickres.first != constrainVerticesRef.cend())
                    idx = pickres.first->first;
            }
        }
        

        if(idx<0){
            ensure(nFace < 16777216, "not implemented for big mesh");

            //glDrawBuffer(GL_BACK);
            glDisable(GL_MULTISAMPLE); // make sure the color will not be interpolated

            //int w = 500, h = 500;
            //glViewport(x-w/2, y-h/2, w, h);
            glClearColor(1.f, 1.f, 1.f, 0.f);
            glClear(GL_COLOR_BUFFER_BIT);
            pickProg.bind();
            pickProg.setUniform("modelview", modelViewMat(vp).data());
            pickProg.setUniform("pickElement", pickElem);  // 0/1 for pick vertex/face
            glBindVertexArray(vaoHandle);

            float pickdist = 12.f;
            //glPointSize(pickdist*mMeshScale);
            glPointSize(pickdist);
            if (pickElem == PE_VERTEX)
                glDrawArrays(GL_POINTS, 0, nVertex);
            else
                glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, nullptr); // not GL_INT

            unsigned char pixel[4];
            glReadPixels(x, vp[3] - y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);   // y is inverted in OpenGL
            //glReadPixels(w/2, h/2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
            idx = (pixel[0] + pixel[1] * 256 + pixel[2] * 256 * 256) - 1;   // -1 to get true index, 0 means background

            glBindVertexArray(0);
            pickProg.unbind();
            //glDrawBuffer(GL_FRONT);
            glClearColor(1.f, 1.f, 1.f, 0.f);
        }


        if (pickElem == PE_VERTEX)
            actVertex = idx;
        else if(pickElem == PE_FACE)
            actFace = idx;

        int res = 0; // return how many vertex are added/deleted
        if (idx>=0){
            //printf("vertex %d is picked\n", idx);
            if (pickElem == PE_VERTEX){
                auto it = constrainVertices.find(idx);
                if (it == constrainVertices.end()){
                    if (operation == PO_ADD && idx < nVertex){  // add constrain
                        vec v;
                        getVertex(idx, v.data());
                        constrainVertices[idx] = v;
                        res = 1;
                    }
                }
                else if (operation == PO_REMOVE){
                    constrainVertices.erase(it);
                    res = -1;
                }
            }
        }

        auto i = getConstrainVertexIds();
        auto it = std::find(i.cbegin(), i.cend(), actVertex);
        actConstrainVertex = int( (it==i.cend())?-1:(it - i.cbegin()) );
        return res;
    }

    void saveResultImage(const char* filename, int width = 3072)
    {
        float edgeWidth0 = edgeWidth;
        float meshScale0 = mMeshScale;
        vec translate = mTranslate;

        auto constrainVtx0 = constrainVertices;
        constrainVertices.clear();

        GLTexture tex1;
        const int vp[] = { 0, 0, width, width };
        tex1.allocateStorage(vp[2], vp[3], GL_RGBA);

        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex1.handle, 0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glReadBuffer(GL_COLOR_ATTACHMENT0);

        GLuint rboDepth;
        glGenRenderbuffers(1, &rboDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, vp[2], vp[3]);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

        glViewport(vp[0], vp[1], vp[2], vp[3]);

        glClearColor(0.f, 0.f, 0.f, 0.f);

        for (int i = 0; i < 1; i++){
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            draw(vp);

            fprintf(stdout, "saving result to %s\n", filename);
            MyImage(tex1).write(filename);
        }

        constrainVertices = constrainVtx0;
        edgeWidth = edgeWidth0;
        mMeshScale = meshScale0;
        mTranslate = translate;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);

        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glDeleteRenderbuffers(1, &rboDepth);

        glClearColor(1.f, 1.f, 1.f, 0.f);
    }

    static void boundingbox(const R* x, int nv, R *bbox)
    {
        if (nv < 1) {
            printf("empty point set!\n");
            return;
        }

        for (int i = 0; i < dim; i++) bbox[i] = bbox[i + dim] = x[i];

        for (int i = 1; i < nv; i++){
            for (int j = 0; j < dim; j++){
                bbox[j] = std::min(bbox[j], x[i*dim + j]);
                bbox[j + dim] = std::max(bbox[j + dim], x[i*dim + j]);
            }
        }
    }

    static void buildShaders()	{
        const char* vertexShaderStr =
            R"( #version 330
    layout (location = 0) in vec2 VertexPosition;
    layout (location = 2) in vec2 VertexTexCoord;
    layout (location = 5) in float VertexData;
    out vec2 TexCoord;
    //uniform vec2 xform, translate;
    //uniform float meshScale;
    uniform mat4 modelview;

    uniform bool colorCoding = false;
    uniform int depthMode = 0;
    uniform vec2 dataMinMax = vec2(0, 1);

    void main(){
        //gl_Position = vec4(meshScale*xform*(VertexPosition+translate), 0, 1);
        gl_Position = modelview*vec4(VertexPosition, 0, 1);
        if(depthMode==0)  gl_Position.z = VertexTexCoord[0]/100; // todo: remove
        else if(depthMode==1)  gl_Position.z = -VertexTexCoord[0]/100; // todo: remove
        else if(depthMode==2)  gl_Position.z = +VertexTexCoord[1]/100; // todo: remove
        else if(depthMode==3)  gl_Position.z = -VertexTexCoord[1]/100; // todo: remove

        if(colorCoding)
            TexCoord = vec2((VertexData-dataMinMax.x)/(dataMinMax.y-dataMinMax.x), 0.5);
        else
            TexCoord = VertexTexCoord;
    })";

        const char*  fragShaderStr =
            R"( #version 330
    in vec2 TexCoord;
    out vec4 FragColor;
    uniform sampler2D img;
    uniform float textureScale;
    uniform vec4 color;
    uniform bool useTexMap;
    void main(){
        FragColor = useTexMap?texture(img, textureScale*TexCoord):color;
        if(FragColor.a<1e-1)
            discard;  // important for transpancy with self overlapping
    })";

        prog.compileAndLinkAllShadersFromString(vertexShaderStr, fragShaderStr);
        prog.bind();
        prog.setUniform("img", 0);
        //prog.setUniform("translate", 0.f, 0.f);
        //prog.setUniform("meshScale", .9f);
        prog.setUniform("textureScale", 1.f);

        pointSetProg.compileAndLinkAllShadersFromString(
R"( #version 330
    layout (location = 0) in vec2 VertexPosition;
    layout (location = 1) in float VertexData;
    uniform float scale;
    uniform mat4 modelview;
    void main(){
        gl_PointSize = VertexData*scale;
        gl_Position = modelview*vec4(VertexPosition, 0, 1);
    })",
R"( #version 330
    out vec4 FragColor;
    uniform vec4 color;
    void main(){ FragColor = color; })");


//////////////////////////////////////////////////////////////////////////
        pickProg.compileAndLinkAllShadersFromString(
R"( #version 330
    layout (location = 0) in vec2 VertexPosition;
    flat out int vertexId;
    uniform mat4 modelview;
    void main(){
        gl_Position = modelview*vec4(VertexPosition, 0, 1);
        vertexId = gl_VertexID;
    })",
R"( #version 330
    uniform int pickElement;
    flat in int vertexId;
    out vec4 FragColor;
    void main(){
        int id = ( (pickElement==0)?vertexId:gl_PrimitiveID ) + 1;
        // Convert the integer id into an RGB color
        FragColor = vec4( (id & 0x000000FF) >>  0, (id & 0x0000FF00) >>  8, (id & 0x00FF0000) >> 16, 255.f)/255.f;
    })");

        colormapTex.setImage(MyImage((BYTE*)jetmaprgb, sizeof(jetmaprgb) / 3, 1, sizeof(jetmaprgb) / 3, 3));
        colormapTex.setClamping(GL_CLAMP_TO_EDGE);
    }
};


typedef GLMesh<> MyMesh;
