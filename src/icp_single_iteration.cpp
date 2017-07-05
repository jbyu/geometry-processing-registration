#include "icp_single_iteration.h"
#include "random_points_on_mesh.h"
#include "point_mesh_distance.h"
#include "point_to_point_rigid_matching.h"
#include "point_to_plane_rigid_matching.h"
#include <iostream>

double icp_single_iteration(
  const Eigen::MatrixXd & VX,
  const Eigen::MatrixXi & FX,
  const Eigen::MatrixXd & VY,
  const Eigen::MatrixXi & FY,
  const int num_samples,
  const ICPMethod method,
  Eigen::Matrix3d & R,
  Eigen::RowVector3d & t)
{
  // Replace with your code
  R = Eigen::Matrix3d::Identity();
  t = Eigen::RowVector3d::Zero();

  Eigen::MatrixXd X, P;

  random_points_on_mesh(num_samples, VX, FX, X);
  Eigen::VectorXd D;
  Eigen::MatrixXd N;
  point_mesh_distance(X, VY, FY, D, P, N);

  double error_before = D.sum();
  //double error_before = (X - P).rowwise().norm().sum();

  switch (method) {
  default:
  case ICP_METHOD_POINT_TO_POINT:
	  point_to_point_rigid_matching(X, P, R, t);
	  break;
  case ICP_METHOD_POINT_TO_PLANE:
	  point_to_plane_rigid_matching(X, P, N, R, t);
	  break;
  }

  // Apply transformation to source mesh
  auto X2 = ((X*R).rowwise() + t).eval();
  point_mesh_distance(X2, VY, FY, D, P, N);
  double error_after = D.sum();
  //double error_after = (X2 - P).rowwise().norm().sum();

  //std::cout << error_before << std::endl;
  //std::cout << error_after << std::endl;
  //std::cout << R << std::endl;
  //std::cout << t << std::endl;
  return error_before - error_after;
}
