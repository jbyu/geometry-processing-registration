#include "random_points_on_mesh.h"
#include "igl/random_points_on_mesh.h"

void random_points_on_mesh(
  const int n,
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & X)
{
	// REPLACE WITH YOUR CODE:
	X.resize(n,3);

#if 1
	Eigen::MatrixXd B;
	Eigen::VectorXi FI;
	igl::random_points_on_mesh(n, V, F, B, FI);
	for (int i = 0; i < n; i++) {
		const int idx = FI[i];
		if (0 > idx) {
			continue;
		}
		const Eigen::VectorXi &face = F.row(idx);
		const Eigen::VectorXd &b = B.row(i);
		const Eigen::VectorXd &v0 = V.row(face[0]);
		const Eigen::VectorXd &v1 = V.row(face[1]);
		const Eigen::VectorXd &v2 = V.row(face[2]);
		X.row(i) = v0 * b[0] + v1*b[1] + v2*b[2];
	}
#else 
	const int size = V.rows();
	for (int i = 0; i < n; i++) {
		int idx = rand() % size;
		X.row(i) = V.row(idx);
	}
#endif
}

