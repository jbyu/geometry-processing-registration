#include "mesh_deformation.h"
#include <igl/random_points_on_mesh.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/arap.h>
#include <igl/colon.h>
#include <algorithm>

//igl::ARAPData *arap_data = nullptr;
igl::ARAPData arap_data;
Eigen::VectorXi S;
Eigen::VectorXi b;

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
	const int num_sample = targetVertices.rows();
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
#if 1
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

