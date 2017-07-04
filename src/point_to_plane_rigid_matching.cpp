#include "point_to_plane_rigid_matching.h"
#include <igl/fit_rotations.h>
#include <igl/polar_svd3x3.h>

void point_to_plane_rigid_matching(
  const Eigen::MatrixXd & X,
  const Eigen::MatrixXd & P,
  const Eigen::MatrixXd & N,
  Eigen::Matrix3d & R,
  Eigen::RowVector3d & t)
{
  // Replace with your code
  R = Eigen::Matrix3d::Identity();
  t = Eigen::RowVector3d::Zero();
  
  const int size = X.rows();
  Eigen::MatrixXd A(size,6);
  Eigen::VectorXd B(size);

  for (int i = 0; i < size; ++i) {
	  const Eigen::RowVector3d& normal = N.row(i);
	  const Eigen::RowVector3d& src = X.row(i);
	  const Eigen::RowVector3d& dest = P.row(i);
	  A.row(i) << normal.cross(src), normal;
	  B[i] = (dest - src).dot(normal);
  }

  Eigen::MatrixXd A_ = A.transpose()*A;
  Eigen::MatrixXd B_ = A.transpose()*B;
  Eigen::MatrixXd opt = A_.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B_);

  Eigen::Matrix3d R_;
  R_(0, 0) = 1;
  R_(0, 1) =-opt(2, 0);
  R_(0, 2) = opt(1, 0);

  R_(1, 0) = opt(2, 0);
  R_(1, 1) = 1;
  R_(1, 2) =-opt(0, 0);

  R_(2, 0) =-opt(1, 0);
  R_(2, 1) = opt(0, 0);
  R_(2, 2) = 1;

  igl::polar_svd3x3(R_, R);

  t(0) = opt(3);
  t(1) = opt(4);
  t(2) = opt(5);
}
