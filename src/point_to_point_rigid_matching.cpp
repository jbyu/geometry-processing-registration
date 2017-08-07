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
#if 0
	igl::fit_rotations(S, true, R);
#else
	typedef Eigen::Matrix<double, 3, 3> Mat3;
	typedef Eigen::Matrix<double, 3, 1> Vec3;
	Mat3 ri, ti, ui, vi;
	Vec3 singular;
	igl::polar_svd(S, ri, ti, ui, singular, vi);
	assert(ri.determinant() >= 0);
	R = ri.transpose();
	//std::cout << singular << std::endl << std::endl;
#endif

	t = cp - cx * R;
}

