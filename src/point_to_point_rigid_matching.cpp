#include "point_to_point_rigid_matching.h"
#include <igl/fit_rotations.h>

void point_to_point_rigid_matching(
	const Eigen::MatrixXd & X,
	const Eigen::MatrixXd & P,
	Eigen::Matrix3d & R,
	Eigen::RowVector3d & t)
{
	// Replace with your code
	R = Eigen::Matrix3d::Identity();
	t = Eigen::RowVector3d::Zero();

	auto cx = X.colwise().mean();
	auto cp = P.colwise().mean();
	
	Eigen::MatrixXd ox = X;
	int sizeX = ox.rows();
	for (int i = 0; i < sizeX; ++i) {
		ox.row(i) -= cx;
	}

	Eigen::MatrixXd op = P;
	int sizeP = op.rows();
	for (int i = 0; i < sizeP; ++i) {
		op.row(i) -= cp;
	}

	Eigen::MatrixXd S = op.transpose()*ox;
	igl::fit_rotations(S, true, R);
	t = cp - cx * R;
}

