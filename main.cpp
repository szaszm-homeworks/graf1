//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

const float EPSILON = 0.000001;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

const float PI = 3.14156265358979f;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, const char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4 MVP;				// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	// Attrib Array 1
	out vec3 color;					// output attribute

	void main() {
		color = vertexColor;			// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;	// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
	precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;			// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	constexpr mat4()
		:mat4(1.0f, 0.0f, 0.0f, 0.0f,
		      0.0f, 1.0f, 0.0f, 0.0f,
		      0.0f, 0.0f, 1.0f, 0.0f,
		      0.0f, 0.0f, 0.0f, 1.0f)
	{ }

	constexpr mat4(float m00, float m01, float m02, float m03,
	               float m10, float m11, float m12, float m13,
	               float m20, float m21, float m22, float m23,
	               float m30, float m31, float m32, float m33) 
		:m { m00, m01, m02, m03,
		     m10, m11, m12, m13,
		     m20, m21, m22, m23,
		     m30, m31, m32, m33 }
	{ }

	mat4(const mat4 &m) :mat4(m[0][0], m[0][1], m[0][2], m[0][3],
	                          m[1][0], m[1][1], m[1][2], m[1][3],
				  m[2][0], m[2][1], m[2][2], m[2][3],
				  m[3][0], m[3][1], m[3][2], m[3][3])
	{ }

	mat4 operator*(const mat4& right) const {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	operator float*() { return &m[0][0]; }
	operator const float*() const { return &m[0][0]; }

	float *operator[](unsigned int i) { return &m[i][0]; }
	const float *operator[](unsigned int i) const { return &m[i][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	explicit vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4(const vec4 &vec) {
		for(int i = 0; i < 4; ++i) {
			v[i] = vec[i];
		}
	}

	vec4 operator*(const mat4& mat) const {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	float operator[](size_t index) const {
		return v[index];
	}

	float &operator[](size_t index) {
		return v[index];
	}

	vec4 &operator+=(const vec4 &w) {
		for(int i = 0; i < 4; ++i) {
			v[i] += w[i];
		}
		return *this;
	}

	vec4 operator+(const vec4 &w) const {
		return (vec4(*this) += w);
	}

	vec4 &operator*=(float f) {
		for(int i = 0; i < 4; ++i) v[i] *= f;
		return *this;
	}

	vec4 operator*(float f) const {
		return (vec4(*this) *= f);
	}

	friend vec4 operator*(float f, const vec4 &v) {
		return v * f;
	}

	vec4 &operator-=(const vec4 &w) {
		return *this += -1.0f * w;
	}

	vec4 operator-(const vec4 &w) const {
		return vec4(*this) -= w;
	}

	vec4 operator-() const {
		return -1.0f * *this;
	}

	vec4 &operator/=(float f) {
		if(f == 1) return *this;
		return *this *= (1.0f / f);
	}

	vec4 operator/(float f) const {
		if(f == 1) return *this;
		return vec4(*this) /= f;
	}

	vec4 &normalize() {
		if(fabs(v[3] - 1.0f) > EPSILON) {
			for(int i = 0; i < 3; ++i) {
				v[i] /= v[3];
			}
			v[3] = 1.0f;
		}

		float len = sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2));

		for(int i = 0; i < 3; ++i) {
			v[i] /= len;
		}
		return *this;
	}

	vec4 getNormalized() const {
		return vec4(*this).normalize();
	}
};

float dot(const vec4 &v, const vec4 &w) {
	float sum = 0.0f;

	if(fabs(v[3] - 1.0f) > EPSILON || fabs(w[3] - 1.0f) > EPSILON) {
		return -1.0f;
	}

	for(int i = 0; i < 3; ++i) {
		sum += v[i] * w[i];
	}

	return sum;
}

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	Camera(const Camera &) = delete;

	mat4 V() const { // view matrix: translates the center to the origin
		return mat4(    1,    0, 0, 0,
			        0,    1, 0, 0,
			        0,    0, 1, 0,
			     -wCx, -wCy, 0, 1);
	}

	mat4 P() const { // projection matrix: scales it to be a square of edge length 2
		return mat4(2/wWx,        0, 0, 0,
			        0,    2/wWy, 0, 0,
			        0,        0, 1, 0,
			        0,        0, 0, 1);
	}

	mat4 Vinv() const { // inverse view matrix
		return mat4(    1,     0, 0, 0,
				0,     1, 0, 0,
			        0,     0, 1, 0,
			        wCx, wCy, 0, 1);
	}

	mat4 Pinv() const { // inverse projection matrix
		return mat4(wWx/2, 0,    0, 0,
			        0, wWy/2, 0, 0,
			        0,  0,    1, 0,
			        0,  0,    0, 1);
	}

	void Animate(float /*t*/) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 2;
		wWy = 2;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
protected:
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	float rotation;		// rotation
public:
	Triangle() {
		Animate(0);
	}

	Triangle(const Triangle &) = delete;

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -0.08f, -0.08f, -0.06f, 0.1f, 0.08f, -0.02f };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			         sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0); 
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			                  2, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

		// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float /*t*/) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
		wTx = 0; // 4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
	}

	void Draw() const {
		mat4 Mscale(sx,  0, 0, 0,
		             0, sy, 0, 0,
			     0,  0, 0, 0,
			     0,  0, 0, 1); // model matrix

		mat4 Mrotate(   cos(rotation), -sin(rotation), 0, 0,
				sin(rotation),  cos(rotation), 0, 0,
				            0,              0, 1, 0,
					    0,              0, 0, 1);

		mat4 Mtranslate(  1,   0,  0, 0,
			          0,   1,  0, 0,
			          0,   0,  0, 0,
			        wTx, wTy,  0, 1); // model matrix

		mat4 MVPTransform = Mscale * Mrotate * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

class LineStrip {
protected:
	static const unsigned int ELEMENTS_PER_VERTEX = 5;
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	std::vector<float> vertexData; // interleaved data of coordinates and colors

	void CopyVertexDataToGPU() const {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_DYNAMIC_DRAW);
	}
public:
	LineStrip() = default;
	LineStrip(const LineStrip &) = delete;

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, ELEMENTS_PER_VERTEX * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, ELEMENTS_PER_VERTEX * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY, float r = 1.0f, float g = 1.0f, float b = 1.0f) {

		vec4 wVertex = vec4(cX, cY, 0, 1);
		// fill interleaved data
		vertexData.push_back(wVertex.v[0]);
		vertexData.push_back(wVertex.v[1]);
		vertexData.push_back(r); // red
		vertexData.push_back(g); // green
		vertexData.push_back(b); // blue
	}

	void Draw() const {
		if (vertexData.size() > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, vertexData.size() / ELEMENTS_PER_VERTEX);
		}
	}

	virtual ~LineStrip() {}
};



class BezierField {
	constexpr static const float CONTROL_POINTS[16] = {
		0.2f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.8f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.3f,
		0.0f, 0.0f, 0.3f, 0.0f,
	};
	static const unsigned int CONTROL_POINTS_WIDTH = 3;

	static const unsigned int GRID_RESOLUTION = 20;
	static const unsigned int ELEMENTS_PER_VERTEX = 5;


	GLuint vao, vbo;
	std::vector<vec4> cps;
	std::vector<float> vertexData;
	float minHeight, maxHeight;

	void CopyVertexDataToGPU() const {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);
	}

	float getBBinom(int i) const {
		int n = CONTROL_POINTS_WIDTH;
		float choose = 1.0f;
		for(int j = 1; j <= i; ++j) {
			choose += static_cast<float>(n-j+1) / j;
		}
		return choose;
	}

	float B(int i, float t) const {
		int n = CONTROL_POINTS_WIDTH;
		return getBBinom(i) * pow(t, i) * pow(1 - t, n - i);
	}

	float Bderiv(int i, float t) const {
		int n = CONTROL_POINTS_WIDTH + 1;
		float BBinom = getBBinom(i);
		return BBinom * (pow(t, i - 1) * pow(1 - t, n - i) * i - pow(t, i) * (n - i) * pow(1 - t, n - i - 1));
	}


	vec4 getGradient(float x, float y) const {
		vec4 result;
		for(unsigned int v = 0; v <= CONTROL_POINTS_WIDTH; ++v) {
			for(unsigned int u = 0; u <= CONTROL_POINTS_WIDTH; ++u) {
				result[0] += Bderiv(u, x) * B(v, y) * CONTROL_POINTS[v * (CONTROL_POINTS_WIDTH + 1) + u];
				result[1] += B(u, x) * Bderiv(v, y) * CONTROL_POINTS[v * (CONTROL_POINTS_WIDTH + 1) + u];
			}
		}

		return result;
	}

	vec4 getColorByLevel(float height) const {
		float r = 1.0f - pow(2.0f * height - 1.0f, 2.0f) * 0.8f;
		float g = (1.0f - height * 0.92f) * 0.8f;
		return vec4(r, g, 0.0f);
	}

	void tesselate() {
		vertexData.clear();
		float tempVertexData[(GRID_RESOLUTION + 1) * (GRID_RESOLUTION + 1) * ELEMENTS_PER_VERTEX];
		std::vector<unsigned int> tempIndexData;
		float step = 1.0f / GRID_RESOLUTION;

		minHeight = maxHeight = getHeight(0.0f, 0.0f);
		for(unsigned int i = 0; i <= GRID_RESOLUTION; ++i) {
			for(unsigned int j = 0; j <= GRID_RESOLUTION; ++j) {
				float v = i * step;
				float u = j * step;
				float height = getHeight(u, v);
				float x = u;
				float y = v;
				tempVertexData[(i * (GRID_RESOLUTION + 1) + j) * ELEMENTS_PER_VERTEX] = x;
				tempVertexData[(i * (GRID_RESOLUTION + 1) + j) * ELEMENTS_PER_VERTEX + 1] = y;
				tempVertexData[(i * (GRID_RESOLUTION + 1) + j) * ELEMENTS_PER_VERTEX + 2] = height; // temporary
				if(height < minHeight) minHeight = height;
				if(height > maxHeight) maxHeight = height;
			}
		}

		for(unsigned int i = 0; i <= GRID_RESOLUTION; ++i) {
			for(unsigned int j = 0; j <= GRID_RESOLUTION; ++j) {
				float height = tempVertexData[(i * (GRID_RESOLUTION + 1) + j) * ELEMENTS_PER_VERTEX + 2];
				vec4 color = getColorByLevel((height - minHeight) / (maxHeight - minHeight));
				tempVertexData[(i * (GRID_RESOLUTION + 1) + j) * ELEMENTS_PER_VERTEX + 2] = color[0];
				tempVertexData[(i * (GRID_RESOLUTION + 1) + j) * ELEMENTS_PER_VERTEX + 3] = color[1];
				tempVertexData[(i * (GRID_RESOLUTION + 1) + j) * ELEMENTS_PER_VERTEX + 4] = color[2];
			}
		}


		for(unsigned int v = 0; v < GRID_RESOLUTION; ++v) {
			for(unsigned int u = 0; u < GRID_RESOLUTION; ++u) {
				tempIndexData.push_back(u + v * (GRID_RESOLUTION + 1));
				tempIndexData.push_back(u + (v + 1) * (GRID_RESOLUTION + 1));
				tempIndexData.push_back(u + 1 + v * (GRID_RESOLUTION + 1));
				tempIndexData.push_back(u + (v + 1) * (GRID_RESOLUTION + 1));
				tempIndexData.push_back(u + 1 + v * (GRID_RESOLUTION + 1));
				tempIndexData.push_back(u + 1 + (v + 1) * (GRID_RESOLUTION + 1));
			}
		}

		for(unsigned int index: tempIndexData) {
			for(unsigned int i = index * ELEMENTS_PER_VERTEX; i < (index + 1) * ELEMENTS_PER_VERTEX; ++i) {
				vertexData.push_back(tempVertexData[i]);
			}
		}
		fflush(stdout);

		CopyVertexDataToGPU();
	}


public:
	float getHeight(float x, float y) const {
		float height = 0.0f;
		for(unsigned int v = 0; v <= CONTROL_POINTS_WIDTH; ++v) {
			for(unsigned int u = 0; u <= CONTROL_POINTS_WIDTH; ++u) {
				height += B(u, x) * B(v, y) * CONTROL_POINTS[v * (CONTROL_POINTS_WIDTH + 1) + u];
			}
		}
		return height;
	}

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, ELEMENTS_PER_VERTEX * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, ELEMENTS_PER_VERTEX * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		tesselate();
	}

	void Draw() const {
		mat4 scale = {
			 2.0f,  0.0f,  0.0f,  0.0f,
			 0.0f,  2.0f,  0.0f,  0.0f,
			 0.0f,  0.0f,  2.0f,  0.0f,
			 0.0f,  0.0f,  0.0f,  1.0f
		};
		mat4 translate = {
			  1.0f,   0.0f,  0.0f,  0.0f,
			  0.0f,   1.0f,  0.0f,  0.0f,
			  0.0f,   0.0f,  1.0f,  0.0f,
			 -1.0f,  -1.0f,  0.0f,  1.0f
		};
		mat4 VPTransform = scale * translate * camera.V() * camera.P();


		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, vertexData.size() / ELEMENTS_PER_VERTEX);
	}

	float getDirectionalDerivative(vec4 position, vec4 direction) const {
		position /= 2.0f;
		position += vec4(0.5f, 0.5f);
		vec4 grad = getGradient(position[0], position[1]);
		float dotproduct = dot(grad, direction);
		return dotproduct;
	}
};

constexpr const float BezierField::CONTROL_POINTS[16];

class LagrangeCurve : protected LineStrip {
	static const int RESOLUTION = 100;
	std::vector<vec4> cps;
	std::vector<float> ts;
	float lastAbsoluteTime = -1.0f;
	float lastRelativeTime = -1.0f; // ~= lastAbsoluteTime - firstAbsoluteTime
	float firstAbsoluteTime = -1.0f;
	const BezierField *field;

	float L(unsigned int i, float t) const {
		float Li = 1.0f;
		for(unsigned int j = 0; j < cps.size(); ++j) {
			if(j == i) continue;
			Li *= (t - ts[j]) / (ts[i] - ts[j]);
		}
		return Li;
	}

	float Lderiv(unsigned int i, float t) const {
		float sum = 0.0f;
		for(unsigned int j = 0; j < cps.size(); ++j) {
			if(j == i) continue;
			sum += 1.0f / (t - ts[j]);
		}
		return sum * L(i, t);
	}


	void tesselate() {
		vertexData.clear();

		float step = lastRelativeTime / static_cast<float>(RESOLUTION);
		for(unsigned i = 0; i <= RESOLUTION; ++i) {
			float t = step * i;
			vec4 point = r(t);
			AddPoint(point[0], point[1]);
		}

		CopyVertexDataToGPU();
	}

public:
	LagrangeCurve(const BezierField *field) :field(field) { }

	vec4 r(float t) const {
		// EPSILON is needed to make a difference between the first and the last point even after fmod
		// EPSILON is a very small positive number
		t = fmod(t, lastRelativeTime + EPSILON);
		vec4 rr(0,0,0);
		for(unsigned int i = 0; i < cps.size(); ++i) {
			rr += cps[i] * L(i, t);
		}
		return rr;
	}

	vec4 direction(float t) const {
		if(lastRelativeTime == -1) return vec4();
		t = fmod(t, lastRelativeTime + EPSILON);
		vec4 rr(0,0,0);
		for(unsigned int i = 0; i < cps.size(); ++i) {
			rr += cps[i] * Lderiv(i, t);
		}
		return rr;
	}

	void AddControlPoint(vec4 cp) {
		float currentAbsoluteTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
		if(fabs(lastAbsoluteTime + 1.0f) <= EPSILON) {
			firstAbsoluteTime = lastAbsoluteTime = currentAbsoluteTime;
			lastRelativeTime = 0.0f;
		}
		float timeDelta = currentAbsoluteTime - lastAbsoluteTime;
		lastRelativeTime += timeDelta;
		lastAbsoluteTime = currentAbsoluteTime;
		ts.push_back(lastRelativeTime);
		cps.push_back(cp);

		if(cps.size() > 1) tesselate();

		printf("LagrangeCurve length: %f km\n", getLength() / 2.0f);
		fflush(stdout);
	}


	void Create() {
		LineStrip::Create();
	}

	void Draw() const {
		LineStrip::Draw();
	}

	float getLength() const {
		float length = 0.0f;
		for(unsigned int i = 1; i < vertexData.size() / ELEMENTS_PER_VERTEX; ++i) {
			float curx = vertexData[i * ELEMENTS_PER_VERTEX];
			float cury = vertexData[i * ELEMENTS_PER_VERTEX + 1];
			float curz = field->getHeight(0.5f * curx + 0.5f, 0.5f * cury + 0.5f);
			float lastx = vertexData[(i - 1) * ELEMENTS_PER_VERTEX];
			float lasty = vertexData[(i - 1) * ELEMENTS_PER_VERTEX + 1];
			float lastz = field->getHeight(0.5f * lastx + 0.5f, 0.5f * lasty + 0.5f);
			length += sqrt(pow(curx - lastx, 2.0f) + pow(cury - lasty, 2.0f) + pow(curz - lastz, 2.0f));
		}
		return length;
	}
};


class Bicycle {
	static const unsigned int ELEMENTS_PER_VERTEX = 5;
	const LagrangeCurve *curve;
	const BezierField *field;
	float vertexData[ELEMENTS_PER_VERTEX * 4];
	GLuint vao, vbo;
	float translation[2];
	float rotation;
	float sy;
	float verticalAngle;
	bool started;
	float startTime;

public:
	Bicycle(const LagrangeCurve *curve, const BezierField *field) 
		:curve(curve), field(field), translation { 0.0f, 0.0f }, started(false)
	{ }

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, ELEMENTS_PER_VERTEX * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, ELEMENTS_PER_VERTEX * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		vertexData[0]  = -1.0f; // x
		vertexData[1]  = -1.0f; // y
		vertexData[2]  =  1.0f; // r
		vertexData[3]  =  0.0f; // g
		vertexData[4]  =  1.0f; // b

		vertexData[5]  =  0.0f;
		vertexData[6]  =  1.0f;
		vertexData[7]  =  0.0f;
		vertexData[8]  =  1.0f;
		vertexData[9]  =  1.0f;

		vertexData[10] =  0.0f;
		vertexData[11] =  0.0f;
		vertexData[12] =  0.0f;
		vertexData[13] =  0.0f;
		vertexData[14] =  1.0f;

		vertexData[15] =  1.0f;
		vertexData[16] = -1.0f;
		vertexData[17] =  1.0f;
		vertexData[18] =  0.0f;
		vertexData[19] =  1.0f;

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 20 * sizeof(float), vertexData, GL_STATIC_DRAW);
	}

	void Animate(float t) {
		if(!started) return;

		vec4 pos = curve->r(t - startTime);
		translation[0] = pos[0]; // 4 * cosf(t / 2);
		translation[1] = pos[1]; // 4 * sinf(t / 2);
		vec4 direction = curve->direction(t - startTime).normalize();
		this->rotation = atan2(direction[0], direction[1]);;
		verticalAngle = atan(field->getDirectionalDerivative(pos, direction));
		sy = sin(verticalAngle + PI / 2.0f);
	}

	void Draw() const {
		using std::sin;
		using std::cos;

		if(!started) return;

		constexpr float scale = 0.1f;

		mat4 Mscale( scale,       0.0f,  0.0f, 0.0f,
		              0.0f, sy * scale,  0.0f, 0.0f,
		              0.0f,       0.0f, scale, 0.0f,
		              0.0f,       0.0f,  0.0f, 1.0f);

		mat4 Mtranslate(          1.0f,           0.0f, 0.0f, 0.0f,
		                          0.0f,           1.0f, 0.0f, 0.0f,
		                          0.0f,           0.0f, 1.0f, 0.0f,
		                translation[0], translation[1], 0.0f, 1.0f);

		mat4 Mrotate( cos(rotation), -sin(rotation), 0.0f, 0.0f,
		              sin(rotation),  cos(rotation), 0.0f, 0.0f,
		                       0.0f,           0.0f, 1.0f, 0.0f,
		                       0.0f,           0.0f, 0.0f, 1.0f);

		mat4 VPTransform = Mscale * Mrotate * Mtranslate * camera.V() * camera.P();

		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}

	float getVerticalAngle() const {
		return verticalAngle;
	}

	void start() {
		started = true;
		startTime = static_cast<float>(glutGet(GLUT_ELAPSED_TIME)) / 1000.0f;
	}
};

class DerivativeTriangle {
	static const unsigned int ELEMENTS_PER_VERTEX = 5;

	GLuint vao, vbo;
	float vertexData[ELEMENTS_PER_VERTEX * 3];
	const Bicycle *bicycle;
	float verticalAngle;

public:
	explicit DerivativeTriangle(const Bicycle *bicycle) 
		:bicycle(bicycle)
	{ }

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, ELEMENTS_PER_VERTEX * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, ELEMENTS_PER_VERTEX * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void Animate() {
		verticalAngle = bicycle->getVerticalAngle();
	}

	void Draw() {
		float height = fabs(tan(verticalAngle));
		if(verticalAngle < 0) {
			vertexData[0] = 0.0f; // x
			vertexData[1] = 0.0f; // y
			vertexData[2] = 0.0f; // r
			vertexData[3] = 1.0f; // g
			vertexData[4] = 1.0f; // b

			vertexData[5] =   0.0f; // x
			vertexData[6] = height; // y
			vertexData[7] =   0.0f; // r
			vertexData[8] =   1.0f; // g
			vertexData[9] =   1.0f; // b

			vertexData[10] = 1.0f; // x
			vertexData[11] = 0.0f; // y
			vertexData[12] = 0.0f; // r
			vertexData[13] = 1.0f; // g
			vertexData[14] = 1.0f; // b
		} else {
			vertexData[0] = 0.0f; // x
			vertexData[1] = 0.0f; // y
			vertexData[2] = 0.0f; // r
			vertexData[3] = 1.0f; // g
			vertexData[4] = 1.0f; // b

			vertexData[5] =   1.0f; // x
			vertexData[6] = height; // y
			vertexData[7] =   0.0f; // r
			vertexData[8] =   1.0f; // g
			vertexData[9] =   1.0f; // b

			vertexData[10] = 1.0f; // x
			vertexData[11] = 0.0f; // y
			vertexData[12] = 0.0f; // r
			vertexData[13] = 1.0f; // g
			vertexData[14] = 1.0f; // b
		}

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 15 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);

		mat4 scale = {
			 0.1f,  0.0f,  0.0f,  0.0f,
			 0.0f,  0.1f,  0.0f,  0.0f,
			 0.0f,  0.0f,  0.1f,  0.0f,
			 0.0f,  0.0f,  0.0f,  1.0f
		};
		mat4 translate = {
			  1.0f,   0.0f,  0.0f,  0.0f,
			  0.0f,   1.0f,  0.0f,  0.0f,
			  0.0f,   0.0f,  1.0f,  0.0f,
			  0.7f,  -0.8f,  0.0f,  1.0f
		};
		mat4 VPTransform = scale * translate * camera.V() * camera.P();


		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}
};


// The virtual world: collection of two objects
BezierField field;
LagrangeCurve lagrangeCurve(&field);
Bicycle bicycle(&lagrangeCurve, &field);
DerivativeTriangle dTriangle(&bicycle);


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	lagrangeCurve.Create();
	field.Create();
	bicycle.Create();
	dTriangle.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	field.Draw();
	lagrangeCurve.Draw();
	bicycle.Draw();
	dTriangle.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int /*pX*/, int /*pY*/) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if(key == ' ') {
		bicycle.start();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char /*key*/, int /*pX*/, int /*pY*/) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;

		lagrangeCurve.AddControlPoint(vec4(cX, cY) * camera.Pinv() * camera.Vinv());
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int /*pX*/, int /*pY*/) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	bicycle.Animate(sec);					// animate the bicycle object
	dTriangle.Animate();
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

