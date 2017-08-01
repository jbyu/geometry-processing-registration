#ifndef OPTIMAL_NONRIGID_ICP_H
#define OPTIMAL_NONRIGID_ICP_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/KroneckerProduct>
#include <igl/AABB.h>
#include <igl/diag.h>

class OptimalNonrigidICP
{
	typedef std::pair<int, int> Edge;
	
protected:
	Eigen::MatrixXd vTarget, vTemplate;
	Eigen::MatrixXi fTarget, fTemplate;

	std::vector<Edge> _edges;
	std::vector<float> _weights;

	igl::AABB<Eigen::MatrixXd, 3> _tree;

	Eigen::MatrixXd _vertices, _correspondences;

	Eigen::MatrixXd X;
	Eigen::SparseMatrix<double> D;
	Eigen::VectorXd wVec;
	int _numVertices;

public:
	OptimalNonrigidICP(){}
	~OptimalNonrigidICP(){}

	void init()
	{
		edgesInit();
		verticesInit();
		nearestSearchInit();
	}

	void initCompute()
	{
		correspondencesInit();
		weightsInit();
	}

	void edgesInit()
	{
		_edges.clear();
		for (int i = 0, count = fTemplate.rows(); i < count; ++i)
		{
			auto face = fTemplate.row(i);
			int a = face[0];
			int b = face[1];
			int c = face[2];
			_edges.push_back(Edge(a,b));
			_edges.push_back(Edge(b,c));
			_edges.push_back(Edge(c,a));
		}
	}

	void nearestSearchInit()
	{
		_tree.init(vTarget, fTarget);
	}

	void verticesInit()
	{
		_vertices = vTemplate;
		_correspondences.resize(vTemplate.rows(), 3);

		// Set matrix G (equation (3) in Amberg et al.) 
		float gamma = 1;
		Eigen::SparseMatrix<double> G;
		Eigen::Vector4d g(1, 1, 1, gamma);
		igl::diag(g, G);

		_numVertices = vTemplate.rows();
		int m = _edges.size();
		Eigen::SparseMatrix<double> M(m, _numVertices);
		for (int r = 0; r < m; ++r) {
			const Edge& edge = _edges[r];
			M.insert(r, edge.first) = -1;
			M.insert(r, edge.second) = 1;
		}

		// Precompute kronecker product of M and G
		Eigen::SparseMatrix<double> kron_M_G(M.rows()*G.rows(), M.cols()*G.cols());
		kron_M_G = kroneckerProduct(M, G);

		// Set matrix D (equation (8) in Amberg et al.)
		D = Eigen::SparseMatrix<double>(_numVertices, _numVertices * 4);
		for (int i = 0; i < _numVertices; ++i) {
			auto vtx = vTemplate.row(i);
			D.insert(i, 4 * i) = vtx[0];
			D.insert(i, 4 * i+1) = vtx[1];
			D.insert(i, 4 * i+2) = vtx[2];
			D.insert(i, 4 * i+3) = 1;
		}

		// Set weights vector
		wVec = Eigen::VectorXd::Ones(_numVertices);

		// initialize transformation matrix X with identity matrices
		Eigen::MatrixXd I(4, 3);
		I.block(0, 0, 3, 3).setIdentity();
		X = I.replicate(_numVertices, 1);
	}

	void correspondencesInit()
	{	
		int index;
		Eigen::RowVector3d closest;
		for (int i = 0, c= _vertices.rows(); i < c; i++)
		{
			_tree.squared_distance(vTarget, fTarget, _vertices.row(i), index, closest);
			_correspondences.row(i) = closest;
		}
	}

	void weightsInit()
	{
		const int size = vTemplate.rows();
		_weights.resize( size );
		for (int i = 0; i < size; ++i) {
			_weights[i] = 1.0f;
		}
	}

	int compute(float alpha) {
		// Transform source points by current transformation matrix X
		_vertices = D*X;

		// Determine closest points on target U to transformed source points.
		Eigen::VectorXd sqrD;
		Eigen::VectorXi I;
		Eigen::MatrixXd U;
		_tree.squared_distance(vTarget, fTarget, _vertices, sqrD, I, U);

		// Update weight matrix
		Eigen::SparseMatrix<double> W(_numVertices, _numVertices);
		igl::diag(wVec, W);

		// Specify A and B (See equation (12) from paper)



		return 0;
	}


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

