#include "mesh_deformation.h"
#include <igl/random_points_on_mesh.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/arap.h>
#include <igl/colon.h>
#include <algorithm>

#include <igl/boundary_loop.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/per_vertex_normals.h>
#include <igl/writeDMAT.h>
#include <igl/writeOBJ.h>

//igl::ARAPData *arap_data = nullptr;
igl::ARAPData arap_data;
Eigen::VectorXi S;
Eigen::VectorXi b;

void deform_match(
	Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXi & sourceFaces,
	const Eigen::MatrixXi & source_landmarks,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces,
	const Eigen::MatrixXi & target_landmarks)
{
#if 0
	const int count = source_landmarks.rows();
	Eigen::MatrixXd X;
	X.resize(count, 3);
	b.resize(count);
	for (int i = 0; i < count; ++i) {
		b[i] = source_landmarks(i, 0);
		int idx = target_landmarks(i, 0);
		X.row(i) = targetVertices.row(idx);
	}
#else
	Eigen::VectorXi boundary;
	igl::boundary_loop(targetFaces, boundary);

	Eigen::MatrixXd sourceNormals;
	igl::per_vertex_normals(sourceVertices, sourceFaces, sourceNormals);

	Eigen::MatrixXd targetNormals;
	igl::per_vertex_normals(targetVertices, targetFaces, targetNormals);

	Eigen::VectorXd sqrD;
	Eigen::VectorXi I;
	Eigen::MatrixXd U;
	igl::point_mesh_squared_distance(sourceVertices, targetVertices, targetFaces, sqrD, I, U);

	Eigen::Vector3d m = sourceVertices.colwise().minCoeff();
	Eigen::Vector3d M = sourceVertices.colwise().maxCoeff();
	double threshold = (M - m).squaredNorm() * 0.0625;

	int numVertices = sourceVertices.rows();
	Eigen::VectorXi w = Eigen::VectorXi::Ones(numVertices);

	for (int i = 0; i < numVertices; ++i) {
		bool prune = false;

		if (sqrD[i] > threshold) {
			prune = true;
			w[i] = 0;
			continue;
		}
#if 1
		// avoid boundary sampling
		auto& face = targetFaces.row(I[i]);
		for (int j = 0, c = boundary.size(); j < c; ++j) {
			if (boundary[j] == face[0] ||
				boundary[j] == face[1] ||
				boundary[j] == face[2])
			{
				prune = true;
				w[i] = 0;
				break;
			}
		}
		if (prune)
			continue;
#endif
		auto &source = sourceVertices.row(i);
		Eigen::RowVector3d dir = (U.row(i) - source);
		double distance = dir.norm();
		dir /= distance;

		// avoid weird direction
		auto &normal = sourceNormals.row(i);
		if (abs(normal.dot(dir)) < 0.707) {
			w[i] = 0;
			continue;
		}
		if (0 > normal.dot(targetNormals.row(face[0]))) {
			w[i] = 0;
			continue;
		}
		if (0 > normal.dot(targetNormals.row(face[1]))) {
			w[i] = 0;
			continue;
		}
		if (0 > normal.dot(targetNormals.row(face[2]))) {
			w[i] = 0;
			continue;
		}

		// avoid self intersection
		auto & V = sourceVertices;
		auto & F = sourceFaces;
		for (int f = 0; f < F.rows(); ++f)
		{
			int i0 = F(f, 0);
			int i1 = F(f, 1);
			int i2 = F(f, 2);
			if (i0 == i ||
				i1 == i ||
				i2 == i) {
				continue;
			}
			// Should be but can't be const 
			Eigen::RowVector3d v0 = V.row(i0).template cast<double>();
			Eigen::RowVector3d v1 = V.row(i1).template cast<double>();
			Eigen::RowVector3d v2 = V.row(i2).template cast<double>();
			// shoot ray, record hit
			double t, u, v;
			if (intersect_triangle1(source.data(), dir.data(), v0.data(), v1.data(), v2.data(),
				&t, &u, &v) && t > 0 && t < distance)
			{
				prune = true;
				w[i] = 0;
				break;
			}
		}
	}

	igl::colon<int>(0, numVertices - 1, b);
	b.conservativeResize(std::stable_partition(b.data(), b.data() + b.size(),
		[&](int i)->bool {return w(i) > 0; }) - b.data());

	Eigen::MatrixXd X;
	X.resize(b.size(), 3);
	double alpha = 0.9;
	for (int i = 0; i < b.size(); ++i) {
		int idx = b[i];
		X.row(i) = U.row(idx)*alpha + sourceVertices.row(idx)*(1.f-alpha);
	}
#endif
	// h  dynamics time step
	arap_data.h = 1;
	// ym  ~Young's modulus smaller is softer, larger is more rigid/stiff
	arap_data.ym = 0.1;
	//arap_data.max_iter = 100;
	//arap_data.with_dynamics = true;
	igl::arap_precomputation(sourceVertices, sourceFaces, sourceVertices.cols(), b, arap_data);
	igl::arap_solve(X, arap_data, sourceVertices);

#if 0
	//Kronecker product
	Eigen::SparseMatrix<float> G;
	Eigen::Vector4d D(1, 1, 1, 0.5);
	igl::diag(D, G);

	Eigen::SparseMatrix<float> M(2,2);
	M.insert(0, 0) = 1;
	//M.insert(0, 1) = 2;
	M.insert(1, 0) = -1;
	//M.insert(1, 1) = 4;
	std::cout << M << std::endl << std::endl << G << std::endl << std::endl;

	Eigen::SparseMatrix<float> kron_M_G(M.rows()*G.rows(), M.cols()*G.cols());
	kron_M_G = kroneckerProduct(M, G);

	std::cout << kron_M_G << std::endl;

	Eigen::MatrixXd I = Eigen::MatrixXd::Zero(4, 3);
	I.block(0, 0, 3, 3).setIdentity();
	Eigen::MatrixXd X2 = I.replicate(4, 1);

	std::cout << X2 << std::endl;
#endif
}

#define kThreshold 0.01f
#if 1
void deform_init(
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXi & sourceFaces,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces)
{
	// random sampling on target
#if 0
	const int num_sample = 100;
#else
	const int num_sample = targetVertices.rows()/4;
#endif
	Eigen::MatrixXd B;
	Eigen::VectorXi FI;
	igl::random_points_on_mesh(num_sample, targetVertices, targetFaces, B, FI);

	std::vector<int> vec;
	vec.reserve(FI.size());
	for (int i = 0, c = FI.size(); i < c; ++i) {
		const int fidx = FI[i];
		if (0 > fidx) {
			continue;
		}
		const int idx = targetFaces.row(fidx)[0];
		vec.push_back(idx);
	}

	Eigen::MatrixXd X;
	X = Eigen::MatrixXd::Zero(vec.size(), targetVertices.cols());
	for (int i = 0, c = vec.size(); i < c; ++i) {
		X.row(i) = targetVertices.row(vec[i]);
	}

	// find closet points on source
	Eigen::VectorXd D;
	Eigen::VectorXi I;
	Eigen::MatrixXd C;
	igl::point_mesh_squared_distance(
		X, sourceVertices, sourceFaces,
		D, I, C);

	vec.clear();
	vec.reserve(I.size());
	for (int i = 0, c = I.size(); i < c; ++i) {
		vec.push_back(sourceFaces.row(I[i])[0]);
	}
	//Just using vector, sort + unique
	std::sort(vec.begin(), vec.end());
	vec.erase(unique(vec.begin(), vec.end()), vec.end());

	b.resize(vec.size());
	for (int i = 0, c = vec.size(); i < c; ++i) {
		b[i] = vec[i];
	}
	//std::cout << b << std::endl;

	igl::arap_precomputation(sourceVertices, sourceFaces, sourceVertices.cols(), b, arap_data);
}

float stiffness = 0.5f;
float delta = 0.1f;

void deform_solve(Eigen::MatrixXd & output,
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces)
{
	Eigen::MatrixXd X;
	X = Eigen::MatrixXd::Zero(b.size(), sourceVertices.cols());
	for (int i = 0, c = b.size(); i < c; ++i) {
		X.row(i) = sourceVertices.row(b(i));
	}
	Eigen::VectorXd D;
	Eigen::VectorXi I;
	Eigen::MatrixXd C;
	igl::point_mesh_squared_distance(
		X, targetVertices, targetFaces,
		D, I, C);
#if 0
	C = C * (1-stiffness) + X*stiffness;
	//stiffness -= delta;
	//if (stiffness < 0) stiffness = 0;
#endif
	igl::arap_solve(C, arap_data, output);
}
#else
void deform_init(
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXi & sourceFaces,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces)
{
	//deform_terminate();
	//arap_data = new igl::ARAPData;

	Eigen::MatrixXd P;
	Eigen::VectorXd D;
	Eigen::MatrixXd N;
	point_mesh_distance(sourceVertices, targetVertices, targetFaces, D, P, N);

#if 1
	igl::colon<int>(0, sourceVertices.rows() - 1, b);
	b.conservativeResize(
		std::stable_partition(b.data(), b.data() + b.size(),
			[&D](int i)->bool {
		return (D[i] < kThreshold) && (rand() % 5 < 1);
	}
	) - b.data());

	std::cout << b.size() << std::endl;
	igl::arap_precomputation(sourceVertices, sourceFaces, sourceVertices.cols(), b, arap_data);
#else

	igl::colon<int>(0, sourceVertices.rows() - 1, b);
	b.conservativeResize(
		std::stable_partition(b.data(), b.data() + b.size(), 
			[&D](int i)->bool {
				return (D[i] < kThreshold) ;
			}
		) - b.data() );
	std::cout << b.size();
	igl::arap_precomputation(sourceVertices, sourceFaces, sourceVertices.cols(), b, *arap_data);
#endif
}

void deform_solve(Eigen::MatrixXd & output,
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces)
{
	Eigen::MatrixXd P;
	Eigen::VectorXd D;
	Eigen::MatrixXd N;
	point_mesh_distance(sourceVertices, targetVertices, targetFaces, D, P, N);

#if 1
	Eigen::MatrixXd input(b.size(), sourceVertices.cols());
	const int size = b.size();
	for (int i = 0; i < size; ++i) {
		int idx = b(i);
		input.row(i) = P.row(idx) *0.999f + sourceVertices.row(idx)*0.001f;
	}
	std::cout << input;
	igl::arap_solve(input, arap_data, output);
#else
	const int size = D.size();
	for (int i = 0; i < size; ++i) {
		if (D[i] > kThreshold)
			P.row(i) = sourceVertices.row(i);
	}
	igl::arap_solve(P, arap_data, output);
#endif
}

#endif


void weld_vertices(const Eigen::MatrixXd & sourceVertices, const Eigen::MatrixXi & sourceFaces,
	Eigen::MatrixXd & outVertices, Eigen::MatrixXi & outFaces, Eigen::VectorXi & mapping)
{
	const int size = sourceVertices.rows();

	mapping.resize(size);
	outFaces.conservativeResizeLike(sourceFaces);
	outVertices.conservativeResizeLike(sourceVertices);

	int count = 0;
	for (int i = 0; i < size; ++i) {
		bool unique = true;
		const auto & vi = sourceVertices.row(i);
		for (int j = 0; j < count; ++j) {
			const auto vj = outVertices.row(j);
			if (vj[0] == vi[0] &&
				vj[1] == vi[1] &&
				vj[2] == vi[2] ) 
			{
				mapping[i] = j;
				unique = false;
				break;
			}
		}
		if (unique) {
			mapping[i] = count;
			outVertices.row(count) = vi;
			++count;
		}
	}
	std::cout << "original vertices: " << size << std::endl;
	std::cout << "unique vertices: " << count << std::endl;
	outVertices.conservativeResize(count, sourceVertices.cols());

	for (int i = 0, c = sourceFaces.rows(); i < c; ++i) {
		for (int j = 0; j < 3; ++j) {
			int idx = sourceFaces(i, j);
			outFaces(i, j) = mapping[idx];
		}
	}
}

void saveWithTexcoord(const std::string & filename,
	const Eigen::MatrixXd & templateVertices,
	const Eigen::MatrixXi & templateFaces,
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXi & sourceFaces,
	const Eigen::MatrixXd & sourceTexcoords,
	const Eigen::MatrixXi & sourceTexFaces,
	const Eigen::VectorXi & mapping,
	const Eigen::MatrixXi & originalFaces)
{
	Eigen::VectorXd sqrD;
	Eigen::VectorXi I, fny;
	Eigen::MatrixXd U, tcY, nY;
	const int size = templateVertices.rows();
	tcY.resize(size, 2);
	igl::point_mesh_squared_distance(templateVertices, sourceVertices, sourceFaces, sqrD, I, U);

	Eigen::MatrixXd templateNormals;
	igl::per_vertex_normals(templateVertices, templateFaces, templateNormals);

	double t, u, v;
	for (int i = 0; i < size; ++i) {
		const int idx = I[i];
		const auto & vface = sourceFaces.row(idx);
		const auto & tface = sourceTexFaces.row(idx);
		Eigen::RowVector3d nrm = templateNormals.row(i);
		Eigen::RowVector3d s_d = templateVertices.row(i);
		Eigen::RowVector3d dir_d = (U.row(i) - s_d).normalized();
		Eigen::RowVector3d v0 = sourceVertices.row(vface[0]);
		Eigen::RowVector3d v1 = sourceVertices.row(vface[1]);
		Eigen::RowVector3d v2 = sourceVertices.row(vface[2]);

		if (intersect_triangle1(s_d.data(), dir_d.data(), v0.data(), v1.data(), v2.data(), &t, &u, &v)) {
		}
		else if (intersect_triangle1(s_d.data(), nrm.data(), v0.data(), v1.data(), v2.data(), &t, &u, &v)) {
		}
		else {
			u = 0;
			v = 0;
		}
		tcY.row(i) = sourceTexcoords.row(tface[0])*(1 - u - v) + sourceTexcoords.row(tface[1])*u + sourceTexcoords.row(tface[2])*v;
#if 0
		else {
			std::cout << i << std::endl;
			std::cout << idx << std::endl;
			std::cout << vface << std::endl;
			std::cout << s_d << std::endl;
			std::cout << dir_d << std::endl;
			std::cout << v0 << std::endl;
			std::cout << v1 << std::endl;
			std::cout << v2 << std::endl << std::endl;
		}
#endif
	}
	
	//igl::writeOBJ("weld.obj", templateVertices, templateFaces, nY, fny, tcY, templateFaces);
	const int num = mapping.size();
	Eigen::MatrixXd origVTX;
	Eigen::MatrixXd origTC;
	origVTX.resize(num, 3);
	origTC.resize(num, 2);
	for (int i = 0; i < num; ++i) {
		int idx = mapping[i];
		origVTX.row(i) = templateVertices.row(idx);
		origTC.row(i) = tcY.row(idx);
	}
	igl::writeOBJ(filename, origVTX, originalFaces, nY, fny, origTC, originalFaces);
}
