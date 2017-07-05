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
	
	Eigen::MatrixXd ox = (X.rowwise() - cx).eval();
	Eigen::MatrixXd op = (P.rowwise() - cp).eval();

	Eigen::MatrixXd S = op.transpose()*ox;
	igl::fit_rotations(S, true, R);

	t = cp - cx * R;
}

