#include "mesh_deformation.h"
#include "point_mesh_distance.h"
#include <igl/random_points_on_mesh.h>
#include <igl/arap.h>
#include <igl/colon.h>
#include <algorithm>

//igl::ARAPData *arap_data = nullptr;
igl::ARAPData arap_data;
Eigen::VectorXi S;
Eigen::VectorXi b;

#define kThreshold 0.01f

void deform_terminate() {
	/*
	if (arap_data) {
		delete arap_data;
		arap_data = nullptr;
	}
	*/
}

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

//igl::arap_solve(bc, arap_data, U);
