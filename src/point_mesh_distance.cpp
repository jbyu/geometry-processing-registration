#include "point_mesh_distance.h"
#include "igl/point_mesh_squared_distance.h"
#include <nanoflann.hpp>

namespace nanoflann {
	/// KD-tree adaptor for working with data directly stored in an Eigen Matrix, without duplicating the data storage.
	/// This code is adapted from the KDTreeEigenMatrixAdaptor class of nanoflann.hpp
	template <class MatrixType, int DIM = -1, class Distance = nanoflann::metric_L2, typename IndexType = int>
	struct KDTreeAdaptor {
		typedef KDTreeAdaptor<MatrixType, DIM, Distance> self_t;
		typedef typename MatrixType::Scalar              num_t;
		typedef typename Distance::template traits<num_t, self_t>::distance_t metric_t;
		typedef KDTreeSingleIndexAdaptor< metric_t, self_t, DIM, IndexType>  index_t;
		index_t* index;
		KDTreeAdaptor(const MatrixType &mat, const int leaf_max_size = 10) : m_data_matrix(mat) {
			const size_t dims = mat.cols();
			index = new index_t(dims, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
			index->buildIndex();
		}
		~KDTreeAdaptor() { delete index; }
		const MatrixType &m_data_matrix;
		/// Query for the num_closest closest points to a given point (entered as query_point[0:dim-1]).
		inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq) const {
			nanoflann::KNNResultSet<typename MatrixType::Scalar, IndexType> resultSet(num_closest);
			resultSet.init(out_indices, out_distances_sq);
			index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
		}
		/// Query for the closest points to a given point (entered as query_point[0:dim-1]).
		inline IndexType closest(const num_t *query_point) const {
			IndexType out_indices;
			num_t out_distances_sq;
			query(query_point, 1, &out_indices, &out_distances_sq);
			return out_indices;
		}
		const self_t & derived() const { return *this; }
		self_t & derived() { return *this; }
		inline size_t kdtree_get_point_count() const { return m_data_matrix.cols(); }
		/// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
		inline num_t kdtree_distance(const num_t *p1, const size_t idx_p2, size_t size) const {
			num_t s = 0;
			for (size_t i = 0; i<size; i++) {
				const num_t d = p1[i] - m_data_matrix.coeff(i, idx_p2);
				s += d*d;
			}
			return s;
		}
		/// Returns the dim'th component of the idx'th point in the class:
		inline num_t kdtree_get_pt(const size_t idx, int dim) const {
			return m_data_matrix.coeff(dim, idx);
		}
		/// Optional bounding-box computation: return false to default to a standard bbox computation loop.
		template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
	};
}

void point_mesh_distance(
  const Eigen::MatrixXd & X,
  const Eigen::MatrixXd & VY,
  const Eigen::MatrixXi & FY,
  const Eigen::MatrixXd & NY,
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

#if 1
	Eigen::VectorXi I;
	igl::point_mesh_squared_distance(
	  X, VY, FY,
	  D, I, P);

	for (int i = 0; i < I.rows(); ++i) {
		//auto face = FY.row(I[i]);
		//N.row(i) = (NY.row(face[0]) + NY.row(face[1]) + NY.row(face[2])).normalized();
		N.row(i) = NY.row(I[i]);
	}
#else
  static nanoflann::KDTreeAdaptor<Eigen::MatrixXd, 3, nanoflann::metric_L2_Simple> kdtree(VY);
  for (int i = 0; i < X.rows(); ++i) {
	  int id = kdtree.closest(X.row(i).data());
	  P.row(i) = VY.row(id);
  }
#endif
}
