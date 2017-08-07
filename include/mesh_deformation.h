#ifndef MESH_DEFORMATION_H
#define MESH_DEFORMATION_H
#include <Eigen/Core>

void deform_match(
	Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXi & sourceFaces,
	const Eigen::MatrixXi & source_landmarks,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces,
	const Eigen::MatrixXi & target_landmarks);

void deform_init(
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXi & sourceFaces,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces);

void deform_solve(Eigen::MatrixXd & output,
	const Eigen::MatrixXd & sourceVertices,
	const Eigen::MatrixXd & targetVertices,
	const Eigen::MatrixXi & targetFaces);

void weld_vertices(const Eigen::MatrixXd & sourceVertices, const Eigen::MatrixXi & sourceFaces,
	Eigen::MatrixXd & outVertices, Eigen::MatrixXi & outFaces, Eigen::VectorXi & mapping);

#endif//MESH_DEFORMATION_H

