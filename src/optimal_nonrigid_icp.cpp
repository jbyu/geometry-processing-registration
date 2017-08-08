#include "optimal_nonrigid_icp.h"

#include <vector>
#include <Eigen/IterativeLinearSolvers>
//#include <unsupported/Eigen/KroneckerProduct>

#include <igl/diag.h>
#include <igl/cat.h>
#include <igl/ismember.h>
#include <igl/boundary_loop.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/per_vertex_normals.h>
#include <igl/adjacency_matrix.h>
#include <igl/point_mesh_squared_distance.h>

void OptimalNonrigidICP::init(const Eigen::MatrixXd& vt, const Eigen::MatrixXi& ft,
	const Eigen::MatrixXd& vs, const Eigen::MatrixXi& fs)
{
	vTarget = vt;
	fTarget = ft;

	vTemplate = vs;
	fTemplate = fs;

	// setup bounding box tree
	_tree.init(vTarget, fTarget);

	igl::per_vertex_normals(vt, ft, targetNormals);

	std::cout << "gather boundary." << std::endl;
	//Detect of the boundary vertices
	igl::boundary_loop(fTarget, targetBoundary);
	igl::boundary_loop(fTemplate, templateBoundary);

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
	_numVertices = vTemplate.rows();
	_numEdges = edges.size();

#if 0
	std::cout << "prepare G." << std::endl;
	// Set matrix G (equation (3) in Amberg et al.) 
	double gamma = 1;
	Eigen::SparseMatrix<double> G;
	Eigen::Vector4d g(1, 1, 1, gamma);
	igl::diag(g, G);
	std::cout << G << std::endl;

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
#else
	kron_M_G = Eigen::SparseMatrix<double>(4 * _numEdges, 4 * _numVertices);
	std::vector< Eigen::Triplet<double> > M_G;
	for (int i = 0; i < _numEdges; ++i)
	{
		Edge edge = edges[i];
		int a = edge.first;
		int b = edge.second;

		for (int j = 0; j < 4; j++)
			M_G.push_back(Eigen::Triplet<double>(i * 4 + j, a * 4 + j, -1));

		for (int j = 0; j < 4; j++)
			M_G.push_back(Eigen::Triplet<double>(i * 4 + j, b * 4 + j, 1));
	}
	kron_M_G.setFromTriplets(M_G.begin(), M_G.end());
#endif

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

	// Set weights vector
	wVec = Eigen::VectorXd::Ones(_numVertices);

	std::cout << "prepare X." << std::endl;
	// initialize transformation matrix X with identity matrices
	Eigen::MatrixXd I = Eigen::MatrixXd::Zero(4, 3);
	I.block(0, 0, 3, 3).setIdentity();
	//std::cout << I << std::endl;
	X = I.replicate(_numVertices, 1);

	oldX = 10*X;
	std::cout << "initial completed." << std::endl;
}

inline void TransfromVertices(Eigen::MatrixXd & out, const Eigen::MatrixXd& in, const Eigen::MatrixXd& xf)
{
	for (int i = 0, c = in.rows(); i < c; ++i) {
		out.row(i) = in.row(i) * xf.block(4 * i, 0, 3, 3) + xf.row(4 * i + 3);
	}
}

int OptimalNonrigidICP::compute(float alpha, float epsilon, Eigen::MatrixXd& deformed) {
	const int kMaxIteration = 16;
	int iteration = 0;
	float error = 0;
	bool preserveBoundary = true;

	alpha = 10;
	epsilon = 1;

	Eigen::MatrixXd transformed(_numVertices, 3);
	Eigen::Vector3d m = vTemplate.colwise().minCoeff();
	Eigen::Vector3d M = vTemplate.colwise().maxCoeff();
	double threshold = (M - m).squaredNorm() * 0.0625;

	while (alpha > 0) {
		do {
			std::cout << "================= " << ++iteration << "th iteration" << " =================" << std::endl;
			std::cout << "alpha: " << alpha << std::endl;
			// Transform source points by current transformation matrix X
			TransfromVertices(transformed, vTemplate, X);

			std::cout << "find closest points." << std::endl;
			// Determine closest points on target U to transformed source points.
			Eigen::VectorXd sqrD;
			Eigen::VectorXi I;
			Eigen::MatrixXd U;
			_tree.squared_distance(vTarget, fTarget, transformed, sqrD, I, U);
			//igl::point_mesh_squared_distance(transformed, vTarget, fTarget, sqrD, I, U);
#if 1
			Eigen::MatrixXd sourceNormals;
			igl::per_vertex_normals(transformed, fTemplate, sourceNormals);

			std::cout << "pruning... ";
			int count = 0;
			const float kMissing = 0;
			for (int i = 0; i < _numVertices; ++i) {
				// avoid far away point
				if (sqrD[i] > threshold) {
					wVec[i] = kMissing;
					++count;
					continue;
				}

				// avoid boundary sampling
				wVec[i] = 1;
				bool prune = false;
				auto & face = fTarget.row(I[i]);
				for (int j = 0, c = targetBoundary.size(); j < c; ++j) {
					if (targetBoundary[j] == face[0] ||
						targetBoundary[j] == face[1] ||
						targetBoundary[j] == face[2])
					{
						wVec[i] = kMissing;
						++count;
						prune = true;
						break;
					}
				}
				if (prune)
					continue;

				auto &source = transformed.row(i);
				Eigen::RowVector3d dir = (U.row(i) - source);
				double distance = dir.norm();
				dir /= distance;

				// avoid weird direction
				auto &normal = sourceNormals.row(i);
				if (abs(normal.dot(dir)) < 0.707) {
					wVec[i] = kMissing;
					++count;
					continue;
				}
				if (0 > normal.dot(targetNormals.row(face[0]))) {
					wVec[i] = kMissing;
					++count;
					continue;
				}
				if (0 > normal.dot(targetNormals.row(face[1]))) {
					wVec[i] = kMissing;
					++count;
					continue;
				}
				if (0 > normal.dot(targetNormals.row(face[2]))) {
					wVec[i] = kMissing;
					++count;
					continue;
				}

				// avoid self intersection
				auto & V = transformed;
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
					Eigen::RowVector3d v0 = V.row(i0);
					Eigen::RowVector3d v1 = V.row(i1);
					Eigen::RowVector3d v2 = V.row(i2);
					// shoot ray, record hit
					double t, u, v;
					if (intersect_triangle1(source.data(), dir.data(), v0.data(), v1.data(), v2.data(),
						&t, &u, &v) && t > 0 && t < distance)
					{
						wVec[i] = kMissing;
						++count;
						break;
					}
				}
			}
			std::cout << count << " points" << std::endl;
#endif

			//preserve boundary of template
			if (preserveBoundary) {
				for (int i = 0, c = templateBoundary.size(); i < c; ++i) {
					int idx = templateBoundary[i];
					wVec[idx] = 1;
					U.row(idx) = vTemplate.row(idx);
				}
			}

			std::cout << "update weights." << std::endl;
			// Update weight matrix
			Eigen::SparseMatrix<double> W;
			igl::diag(wVec, W);
#if 1
			std::cout << "compute A." << std::endl;
			// Specify A and B (See equation (12) from paper)
			Eigen::SparseMatrix<double> A;
			Eigen::SparseMatrix<double> stiffness = alpha*kron_M_G;
			Eigen::SparseMatrix<double> distance = W*D;
			igl::cat(1, stiffness, distance, A);

			std::cout << "compute B." << std::endl;
			Eigen::MatrixXd B = Eigen::MatrixXd::Zero(4 * _numEdges + _numVertices, 3);
			for (int i = 0; i < _numVertices; ++i) {
				B.row(4 * _numEdges + i) = U.row(i) * wVec(i);
			}
#else
			Eigen::SparseMatrix<double> A = W*D;
			Eigen::MatrixXd B = Eigen::MatrixXd::Zero(_numVertices, 3);
			for (int i = 0; i < _numVertices; ++i) {
				B.row(i) = U.row(i) * wVec(i);
			}
#endif
			Eigen::SparseMatrix<double> At(A.transpose());
			Eigen::SparseMatrix<double> ATA = At * A;
			Eigen::MatrixXd ATB = At * B;

			Eigen::ConjugateGradient< Eigen::SparseMatrix<double> > solver;
			solver.compute(ATA);
			std::cout << "solver computed ATA." << std::endl;
			if (solver.info() != Eigen::Success)
			{
				std::cerr << "Decomposition failed" << std::endl;
				return 1;
			}
			oldX = X;
			X = solver.solve(ATB);
			std::cout << "X calculated." << std::endl;
			error = (X - oldX).norm();
			std::cout << "Error: " << error << std::endl;
		} while (error >= epsilon  && iteration < kMaxIteration);
		alpha -= 1;
		iteration = 0;
	}

	deformed.resize(_numVertices, 3);
	TransfromVertices(deformed, vTemplate, X);

	return 0;
}