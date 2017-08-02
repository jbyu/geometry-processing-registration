#include "optimal_nonrigid_icp.h"

#include <vector>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/KroneckerProduct>

#include <igl/diag.h>
#include <igl/cat.h>
#include <igl/ismember.h>
#include <igl/boundary_loop.h>
#include <igl/ray_mesh_intersect.h>

void OptimalNonrigidICP::init(const Eigen::MatrixXd& vt, const Eigen::MatrixXi& ft,
	const Eigen::MatrixXd& vs, const Eigen::MatrixXi& fs)
{
	vTarget = vt;
	fTarget = ft;

	vTemplate = vs;
	fTemplate = fs;

	// setup bounding box tree
	_tree.init(vTarget, fTarget);

	std::cout << "gather boundary." << std::endl;
	//Detect of the boundary vertices
	igl::boundary_loop(fTarget, boundary);

	// setup edge info.
	std::vector<Edge> edges;
	for (int i = 0, count = fTemplate.rows(); i < count; ++i)
	{
		auto face = fTemplate.row(i);
		int a = face[0];
		int b = face[1];
		int c = face[2];
		edges.push_back(Edge(a, b));
		edges.push_back(Edge(b, c));
		edges.push_back(Edge(c, a));
	}

	std::cout << "prepare G." << std::endl;
	// Set matrix G (equation (3) in Amberg et al.) 
	double gamma = 1;
	Eigen::SparseMatrix<double> G;
	Eigen::Vector4d g(1, 1, 1, gamma);
	igl::diag(g, G);

	_numVertices = vTemplate.rows();
	_numEdges = edges.size();
	Eigen::SparseMatrix<double> M(_numEdges, _numVertices);
	for (int r = 0; r < _numEdges; ++r) {
		const Edge& edge = edges[r];
		M.insert(r, edge.first) = -1;
		M.insert(r, edge.second) = 1;
	}

	std::cout << "compute kronecker product of M and G." << std::endl;
	// Precompute kronecker product of M and G
	//Eigen::SparseMatrix<double> kron_M_G(M.rows()*G.rows(), M.cols()*G.cols());
	kron_M_G = kroneckerProduct(M, G);

	std::cout << "prepare D." << std::endl;
	// Set matrix D (equation (8) in Amberg et al.)
	D = Eigen::SparseMatrix<double>(_numVertices, _numVertices * 4);
	for (int i = 0; i < _numVertices; ++i) {
		auto vtx = vTemplate.row(i);
		D.insert(i, 4 * i) = vtx[0];
		D.insert(i, 4 * i + 1) = vtx[1];
		D.insert(i, 4 * i + 2) = vtx[2];
		D.insert(i, 4 * i + 3) = 1;
	}

	std::cout << "prepare weights." << std::endl;
	// Set weights vector
	wVec = Eigen::VectorXd::Ones(_numVertices);

	std::cout << "prepare X." << std::endl;
	// initialize transformation matrix X with identity matrices
	Eigen::MatrixXd I(4, 3);
	I.block(0, 0, 3, 3).setIdentity();
	X = I.replicate(_numVertices, 1);

	std::cout << "initial completed." << std::endl;
}

int OptimalNonrigidICP::compute(float alpha, float epsilon, Eigen::MatrixXd& deformed) {
	// set oldX to be very different to X so that norm(X - oldX) is large on first iteration
	Eigen::MatrixXd oldX = 2 * X;
	int iteration = 0;
	float error = 1;
	alpha = 10;
	epsilon = 10000;
	do {
		std::cout << ++iteration << "th iteration:" << std::endl;
		// Transform source points by current transformation matrix X
		_vertices = D*X;

		std::cout << "find closest points." << std::endl;
		// Determine closest points on target U to transformed source points.
		Eigen::VectorXd sqrD;
		Eigen::VectorXi I;
		Eigen::MatrixXd U;
		_tree.squared_distance(vTarget, fTarget, _vertices, sqrD, I, U);

		std::cout << "pruning... ";
		int prune = 0;
		// avoid boundary sampling
		for (int i = 0; i < _numVertices; ++i) {
			auto & face = fTarget.row(I[i]);
			wVec[i] = 1;
			for (int j = 0, c = boundary.size(); j < c; ++j) {
				if (boundary[j] == face[0] ||
					boundary[j] == face[1] ||
					boundary[j] == face[2])
				{
					wVec[i] = 0;
					++prune;
					break;
				}
			}
			if (0 == wVec[i])
				continue;

			auto &source = _vertices.row(i);
			Eigen::RowVector3d dir = (U.row(i) - source);
			double distance = dir.norm();
			dir /= distance;
			// loop over all triangles
			auto & V = _vertices;
			auto & F = fTemplate;
			for (int f = 0; f < F.rows(); ++f)
			{
				int i0 = F(f, 0);
				int i1 = F(f, 1);
				int i2 = F(f, 2);
				if (i0 == i ||
					i1 == i ||
					i2 == i) {
					continue;
				}
				// Should be but can't be const 
				Eigen::RowVector3d v0 = V.row(i0).template cast<double>();
				Eigen::RowVector3d v1 = V.row(i1).template cast<double>();
				Eigen::RowVector3d v2 = V.row(i2).template cast<double>();
				// shoot ray, record hit
				double t, u, v;
				if (intersect_triangle1(source.data(), dir.data(), v0.data(), v1.data(), v2.data(),
					&t, &u, &v) && t > 0 && t < distance)
				{
					wVec[i] = 0;
					++prune;
					break;
				}
			}
		}
		std::cout << prune<< " points" << std::endl;

		std::cout << "update weights." << std::endl;
		// Update weight matrix
		Eigen::SparseMatrix<double> W(_numVertices, _numVertices);
		igl::diag(wVec, W);

		std::cout << "compute A." << std::endl;
		// Specify A and B (See equation (12) from paper)
		Eigen::SparseMatrix<double> A;
		Eigen::SparseMatrix<double> stiffness = kron_M_G*alpha;
		Eigen::SparseMatrix<double> distance = W*D;
		igl::cat(1, stiffness, distance, A);

		std::cout << "compute B." << std::endl;
		Eigen::MatrixXd B(4 * _numEdges + _numVertices, 3);
		for (int i = 0; i < _numVertices; ++i) {
			B.row(4 * _numEdges + i) = U.row(i) * wVec(i);
		}

		Eigen::SparseMatrix<double> At(A.transpose());
		Eigen::SparseMatrix<double> ATA = At * A;
		std::cout << "ATA calculated." << std::endl;
		Eigen::MatrixXd ATB = At * B;
		std::cout << "ATB calculated." << std::endl;

		Eigen::ConjugateGradient< Eigen::SparseMatrix<double> > solver;
		solver.compute(ATA);
		std::cout << "solver computed ATA." << std::endl;
		if (solver.info() != Eigen::Success)
		{
			std::cerr << "Decomposition failed" << std::endl;
			return 1;
		}
		X = solver.solve(ATB);
		std::cout << "X calculated." << std::endl;
		error = (X - oldX).norm();
		std::cout << "Error: " << error << std::endl;
	} while (error >= epsilon);

	deformed = D*X;
	return 0;
}