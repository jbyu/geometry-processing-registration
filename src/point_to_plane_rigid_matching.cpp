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
#if 1
  const int size = X.rows();
  Eigen::MatrixXd A(size, 6);
  Eigen::VectorXd B(size);

  for (int i = 0; i < size; ++i) {
	  const Eigen::RowVector3d& normal = N.row(i);
	  const Eigen::RowVector3d& src = X.row(i);
	  const Eigen::RowVector3d& dest = P.row(i);
	  A.row(i) << (normal.cross(src)), normal;
	  B[i] = (dest-src).dot(normal);
  }

  Eigen::MatrixXd A_ = A.transpose()*A;
  Eigen::MatrixXd B_ = A.transpose()*B;
  //Eigen::MatrixXd opt = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
  Eigen::MatrixXd opt = A_.ldlt().solve(B_);
/*
  std::cout << A_ << std::endl << std::endl;
  std::cout << B_ << std::endl << std::endl;
  std::cout << opt << std::endl << std::endl;
  std::cout << A_*opt << std::endl << std::endl;
*/
  Eigen::Matrix3d R_;
  R_(0, 0) = 1;
  R_(0, 1) =-opt(2);
  R_(0, 2) = opt(1);

  R_(1, 0) = opt(2);
  R_(1, 1) = 1;
  R_(1, 2) =-opt(0);

  R_(2, 0) =-opt(1);
  R_(2, 1) = opt(0);
  R_(2, 2) = 1;

  igl::polar_svd3x3(R_, R);

  t(0) = opt(3);
  t(1) = opt(4);
  t(2) = opt(5);
#else
  typedef Eigen::Matrix<double, 6, 6> Matrix66;
  typedef Eigen::Matrix<double, 6, 1> Vector6;
  typedef Eigen::Block<Matrix66, 3, 3> Block33;

  /// Prepare LHS and RHS
  Matrix66 LHS = Matrix66::Zero();
  Vector6 RHS = Vector6::Zero();
  Block33 TL = LHS.topLeftCorner<3, 3>();
  Block33 TR = LHS.topRightCorner<3, 3>();
  Block33 BR = LHS.bottomRightCorner<3, 3>();
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(X.rows(), 3);

  {
	  for (int i = 0; i<X.rows(); i++) {
		  const Eigen::RowVector3d& src = X.row(i);
		  const Eigen::RowVector3d& normal = N.row(i);

		  C.row(i) = src.cross(normal);
	  }
	  {
		  for (int i = 0; i<X.rows(); i++)
			  TL.selfadjointView<Eigen::Upper>().rankUpdate(C.row(i), 1);

		  for (int i = 0; i<X.rows(); i++)
			  TR += (C.row(i) * N.row(i).transpose());

		  for (int i = 0; i<X.rows(); i++)
			  BR.selfadjointView<Eigen::Upper>().rankUpdate(N.row(i), 1);

		  for (int i = 0; i<C.rows(); i++) {
			  double dist_to_plane = -((X.row(i) - P.row(i)).dot(N.row(i)) - 0)*1;
			  RHS.head<3>() += C.row(i)*dist_to_plane;
			  RHS.tail<3>() += N.row(i)*dist_to_plane;
		  }
	  }
  }
  LHS = LHS.selfadjointView<Eigen::Upper>();
  /// Compute transformation
  Eigen::Affine3d transformation;
  Eigen::LDLT<Matrix66> ldlt(LHS);
  RHS = ldlt.solve(RHS);
  transformation = Eigen::AngleAxisd(RHS(0), Eigen::Vector3d::UnitX()) *
	  Eigen::AngleAxisd(RHS(1), Eigen::Vector3d::UnitY()) *
	  Eigen::AngleAxisd(RHS(2), Eigen::Vector3d::UnitZ());
  transformation.translation() = RHS.tail<3>();
#endif
}
