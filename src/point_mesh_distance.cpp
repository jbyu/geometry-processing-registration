#include "point_mesh_distance.h"
#include "igl/point_mesh_squared_distance.h"

extern Eigen::MatrixXd NFY;
extern Eigen::MatrixXd NVY;

void point_mesh_distance(
  const Eigen::MatrixXd & X,
  const Eigen::MatrixXd & VY,
  const Eigen::MatrixXi & FY,
  Eigen::VectorXd & D,
  Eigen::MatrixXd & P,
  Eigen::MatrixXd & N)
{
  // Replace with your code
  P.resizeLike(X);
  N = Eigen::MatrixXd::Zero(X.rows(),X.cols());

#if 0
  for(int i = 0;i<X.rows();i++) P.row(i) = VY.row(i%VY.rows());
  D = (X-P).rowwise().norm();
#endif

	Eigen::VectorXi I;
	igl::point_mesh_squared_distance(
	  X, VY, FY,
	  D, I, P);

	for (int i = 0; i < I.rows(); ++i) {
		//auto face = FY.row(I[i]);
		//N.row(i) = (NY.row(face[0]) + NY.row(face[1]) + NY.row(face[2])).normalized();
		N.row(i) = NFY.row(I[i]);
	}
}
