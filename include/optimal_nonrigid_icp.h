#ifndef OPTIMAL_NONRIGID_ICP_H
#define OPTIMAL_NONRIGID_ICP_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/AABB.h>

class OptimalNonrigidICP
{
	typedef std::pair<int, int> Edge;
	
protected:
	Eigen::MatrixXd vTarget, vTemplate;
	Eigen::MatrixXi fTarget, fTemplate;
	Eigen::VectorXi boundary;

	igl::AABB<Eigen::MatrixXd, 3> _tree;

	Eigen::MatrixXd _vertices, _correspondences;

	Eigen::MatrixXd X;
	Eigen::SparseMatrix<double> D;
	Eigen::VectorXd wVec;

	Eigen::SparseMatrix<double> kron_M_G;

	int _numVertices;
	int _numEdges;

public:
	OptimalNonrigidICP(){}
	~OptimalNonrigidICP(){}

	void init(const Eigen::MatrixXd& vt, const Eigen::MatrixXi& ft,
		const Eigen::MatrixXd& vs, const Eigen::MatrixXi& fs);

	int compute(float alpha, float epsilon, Eigen::MatrixXd& deformed);

#if 0
	int compute(float alpha, float beta, float gamma)
	{
		//To do nonrigid icp registration

		int n = _vertices->GetNumberOfPoints();
		int m = _edges->size();

		Eigen::SparseMatrix<float> A(4 * m + n, 4 * n);

		std::vector< Eigen::Triplet<float> > alpha_M_G;
		for (int i = 0; i < m; ++i)
		{
			Edge edge = (*_edges)[i];
			int a = edge.first;
			int b = edge.second;

			for (int j = 0; j < 3; j++) alpha_M_G.push_back(Eigen::Triplet<float>(i * 4 + j, a * 4 + j, alpha));
			alpha_M_G.push_back(Eigen::Triplet<float>(i * 4 + 3, a * 4 + 3, alpha * gamma));

			for (int j = 0; j < 3; j++) alpha_M_G.push_back(Eigen::Triplet<float>(i * 4 + j, b * 4 + j, -alpha));
			alpha_M_G.push_back(Eigen::Triplet<float>(i * 4 + 3, b * 4 + 3, -alpha * gamma));
		}
		std::cout << "alpha_M_G calculated!" << std::endl;

		std::vector< Eigen::Triplet<float> > W_D;
		for (int i = 0; i < n; ++i)
		{
			double xyz[3];
			_vertices->GetPoint(i, xyz);

			float weight = (*_weights)[i];

			for (int j = 0; j < 3; ++j) W_D.push_back(Eigen::Triplet<float>(4 * m + i, i * 4 + j, weight * xyz[j]));
			W_D.push_back(Eigen::Triplet<float>(4 * m + i, i * 4 + 3, weight));
		}
		std::cout << "W_D calculated!" << std::endl;

		std::vector< Eigen::Triplet<float> > _A = alpha_M_G;
		_A.insert(_A.end(), W_D.begin(), W_D.end());
		std::cout << "_A calculated!" << std::endl;

		A.setFromTriplets(_A.begin(), _A.end());
		std::cout << "A calculated!" << std::endl;

		Eigen::MatrixX3f B = Eigen::MatrixX3f::Zero(4 * m + n, 3);
		for (int i = 0; i < n; ++i)
		{
			double xyz[3];
			_correspondences->GetPoint(i, xyz);

			float weight = (*_weights)[i];
			for (int j = 0; j < 3; j++) B(4 * m + i, j) = weight * xyz[j];
		}
		std::cout << "B calculated!" << std::endl;

		Eigen::SparseMatrix<float> ATA = Eigen::SparseMatrix<float>(A.transpose()) * A;
		std::cout << "ATA calculated!" << std::endl;
		Eigen::MatrixX3f ATB = Eigen::SparseMatrix<float>(A.transpose()) * B;
		std::cout << "ATB calculated!" << std::endl;

		Eigen::ConjugateGradient< Eigen::SparseMatrix<float> > solver;
		solver.compute(ATA);
		std::cout << "solver computed ATA!" << std::endl;
		if (solver.info() != Eigen::Success)
		{
			std::cerr << "Decomposition failed" << std::endl;
			return 1;
		}

		Eigen::MatrixX3f X = solver.solve(ATB);
		std::cout << "X calculated!" << std::endl;

		Eigen::Matrix3Xf XT = X.transpose();
		for (int i = 0; i < n; ++i)
		{
			double xyz[3];
			_vertices->GetPoint(i, xyz);
			Eigen::Vector4f point(xyz[0], xyz[1], xyz[2], 1.0f);
			Eigen::Vector3f point_transformed = XT.block<3, 4>(0, 4 * i) * point;
			_vertices->SetPoint(i, point_transformed[0], point_transformed[1], point_transformed[2]);

			double txyz[3];
			_correspondences->GetPoint(i, txyz);

			//if ( i < 10) std::cout << XT.block<3, 4>(0, 4*i) << std::endl;
			if (i < 10) std::cout << xyz[0] << "," << xyz[1] << "," << xyz[2] << " -> "
				<< point_transformed[0] << " " << point_transformed[1] << " " << point_transformed[2] << " -> "
				<< txyz[0] << "," << txyz[1] << "," << txyz[2] << std::endl;

		}

		return 0;
	}
#endif
};

#endif//OPTIMAL_NONRIGID_ICP_H

