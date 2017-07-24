#ifndef MESH_DEFORMATION_H
#define MESH_DEFORMATION_H
#include <Eigen/Core>

void deform_terminate();

void deform_init(
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXi & sourceFaces,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces);

void deform_solve(Eigen::MatrixXd & output,
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces);

#endif//MESH_DEFORMATION_H

