#include "point_triangle_distance.h"
#include "igl/point_simplex_squared_distance.h"

void point_triangle_distance(
  const Eigen::RowVector3d & x,
  const Eigen::RowVector3d & a,
  const Eigen::RowVector3d & b,
  const Eigen::RowVector3d & c,
  double & d,
  Eigen::RowVector3d & p)
{
	Eigen::MatrixXd V, B;
	V.resize(3, 3);
	V.row(0) = a;
	V.row(1) = b;
	V.row(2) = c;
	Eigen::MatrixXi F;
	F.resize(1, 3);
	F.row(0) = Eigen::Vector3i(0, 1, 2);

	//igl::point_simplex_squared_distance(x, V, F, 0, d, p, B);
}
