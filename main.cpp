#include "hausdorff_lower_bound.h"
#include "icp_single_iteration.h"
#include "random_points_on_mesh.h"
#include "point_mesh_distance.h"
#include "point_to_point_rigid_matching.h"

#include <igl/read_triangle_mesh.h>
#include <igl/viewer/Viewer.h>
#include <igl/readDMAT.h>

#include <Eigen/Core>
#include <string>
#include <iostream>

#include "mesh_deformation.h"

Eigen::MatrixXd NFY;
Eigen::MatrixXd NVY;

int main(int argc, char *argv[])
{
  srand(time(0));

  // Load input meshes
  Eigen::MatrixXd OVX,VX,VY,OVY;
  Eigen::MatrixXi FX,FY;
  igl::read_triangle_mesh(
	(argc>1 ? argv[1] : "../data/max-registration-simple.obj"), OVX, FX);
  igl::read_triangle_mesh(
    (argc>2 ? argv[2] : "../data/max-registration-complete.obj"), VY, FY);

  Eigen::MatrixXi I;
  Eigen::MatrixXd target_landmarks;
  igl::read_triangle_mesh(
	  (argc>3 ? argv[3] : "../data/male_feature.obj"), target_landmarks, I);

  Eigen::MatrixXd template_landmarks;
  igl::readDMAT("../data/head.dmat", I);
  template_landmarks.resize(I.rows(), 3);
  for (int i = 0, c = I.rows(); i < c; ++i) {
	  template_landmarks.row(i) = VY.row(I.row(i)[0]);
  }

  igl::per_face_normals(VY, FY, NFY);
  igl::per_vertex_normals(VY, FY, NVY);

#if 1
  // Find the bounding box and normalize data
  Eigen::Vector3d m = VY.colwise().minCoeff();
  Eigen::Vector3d M = VY.colwise().maxCoeff();
  Eigen::Vector3d c = (m + M)*0.5f;
  float scale = (M[0] - m[0])*0.9;

  m = OVX.colwise().minCoeff();
  M = OVX.colwise().maxCoeff();
  float inv_scale = scale / (M[0] - m[0]);
  Eigen::RowVector3d offset(c[0], c[1], c[2]+1);
  OVX *= inv_scale;
  OVX.rowwise() += offset;

#else
	// Find the bounding box and normalize data
	Eigen::Vector3d m = template_landmarks.colwise().minCoeff();
	Eigen::Vector3d M = template_landmarks.colwise().maxCoeff();
	Eigen::Vector3d c = (m + M)*0.5f;
	float scale = (M[0] - m[0])*0.9;

	m = target_landmarks.colwise().minCoeff();
	M = target_landmarks.colwise().maxCoeff();
	float inv_scale = scale / (M[0] - m[0]);
	Eigen::RowVector3d offset(c[0], c[1], c[2] + 1);

	OVX *= inv_scale;
	OVX.rowwise() += offset;
#endif

#if 0
	target_landmarks *= inv_scale;
	target_landmarks.rowwise() += offset;

	// align landmarks
	Eigen::Matrix3d R;
	Eigen::RowVector3d  t;
	//for (int i = 0; i < 100; ++i) 
	{
		point_to_point_rigid_matching(target_landmarks, template_landmarks, R, t);
		target_landmarks = ((target_landmarks*R).rowwise() + t).eval();
		OVX = ((OVX*R).rowwise() + t).eval();
	}
#endif

	//backup vertices
	OVY = VY;

  const int max_iteration = 32;
  int num_iteration = 0;
  int num_samples = 100;
  bool show_samples = true;
  ICPMethod method = ICP_METHOD_POINT_TO_PLANE;

  igl::viewer::Viewer viewer;
  std::cout<<R"(
  [space]  toggle animation
  H,h      print lower bound on directed Hausdorff distance from X to Y
  M,m      toggle between point-to-point and point-to-plane methods
  P,p      show sample points
  R,r      reset, also recomputes a random sampling and closest points
  S        double number of samples
  s        halve number of samples
)";

  // predefined colors
  const Eigen::RowVector3d orange(1.0,0.7,0.2);
  const Eigen::RowVector3d blue(0.2,0.3,0.8);
  const auto & set_meshes = [&](int mode = 0)
  {
	  viewer.data.clear();
	  switch (mode) {
	  default:
	  case 0:
	  {
		  // Concatenate meshes into one big mesh
		  Eigen::MatrixXd V(VX.rows() + VY.rows(), VX.cols());
		  V << VX, VY;
		  Eigen::MatrixXi F(FX.rows() + FY.rows(), FX.cols());
		  F << FX, FY.array() + VX.rows();
		  viewer.data.set_mesh(V, F);
		  // Assign orange and blue colors to each mesh's faces
		  Eigen::MatrixXd C(F.rows(), 3);
		  C.topLeftCorner(FX.rows(), 3).rowwise() = orange;
		  C.bottomLeftCorner(FY.rows(), 3).rowwise() = blue;
		  viewer.data.set_colors(C);
	  }
		  break;
	  case 1:
	  {
		  viewer.data.set_mesh(VX, FX);
		  Eigen::MatrixXd C(FX.rows(), 3);
		  C.rowwise() = orange;
		  viewer.data.set_colors(C);
	  }
		  break;
	  case 2:
	  {
		  viewer.data.set_mesh(VY, FY);
		  Eigen::MatrixXd C(FY.rows(), 3);
		  C.rowwise() = blue;
		  viewer.data.set_colors(C);
	  }
	  break;
	  }
  };
  const auto & set_points = [&]()
  {
    Eigen::MatrixXd X,P;
    random_points_on_mesh(num_samples,VX,FX,X);
    Eigen::VectorXd D;
    Eigen::MatrixXd N;
    point_mesh_distance(X,VY,FY,D,P,N);
    Eigen::MatrixXd XP(X.rows()+P.rows(),3);
    XP<<X,P;
    Eigen::MatrixXd C(XP.rows(),3);
    C.array().topRows(X.rows()).rowwise() = (1.-(1.-orange.array())*.8);
    C.array().bottomRows(P.rows()).rowwise() = (1.-(1.-blue.array())*.4);
    viewer.data.set_points(XP,C);
    Eigen::MatrixXi E(X.rows(),2);
    E.col(0) = Eigen::VectorXi::LinSpaced(X.rows(),0,X.rows()-1);
    E.col(1) = Eigen::VectorXi::LinSpaced(X.rows(),X.rows(),2*X.rows()-1);
    viewer.data.set_edges(XP,E,Eigen::RowVector3d(0.3,0.3,0.3));
  };
  const auto & reset = [&]()
  {
    VX = OVX;
	VY = OVY;
    set_meshes();
	if (show_samples)
	{
		set_points();
	}
  };
  viewer.callback_pre_draw = [&](igl::viewer::Viewer &)->bool
  {
    if(viewer.core.is_animating)
    {
      ////////////////////////////////////////////////////////////////////////
      // Perform single iteration of ICP method
      ////////////////////////////////////////////////////////////////////////
      Eigen::Matrix3d R;
      Eigen::RowVector3d t;
      double delta = icp_single_iteration(VX,FX,VY,FY,num_samples,method,R,t);

      // Apply transformation to source mesh
      VX = ((VX*R).rowwise() + t).eval();

	  set_meshes();
      if(show_samples)
      {
        set_points();
      }
#if 1
	  ++num_iteration;
	  viewer.core.is_animating = (1e-3 < abs(delta)) && (max_iteration > num_iteration);
	  std::cout << num_iteration <<":"<< delta << std::endl;
	  //viewer.core.is_animating = false;
#endif
    }
    return false;
  };
  viewer.callback_key_pressed = 
    [&](igl::viewer::Viewer &,unsigned char key,int)->bool
  {
    switch(key)
    {
      case ' ':
		num_iteration = 0;
        viewer.core.is_animating ^= 1;
        break;
      case 'H':
      case 'h':
        std::cout<<"D_{H}(X -> Y) >= "<<
          hausdorff_lower_bound(VX,FX,VY,FY,num_samples)<<std::endl;
        break;
      case 'M':
      case 'm':
        method = (ICPMethod)((((int)method)+1)%((int)NUM_ICP_METHODS));
        std::cout<< "point-to-"<<
          (method==ICP_METHOD_POINT_TO_PLANE?"plane":"point")<<std::endl;
        break;
      case 'P':
      case 'p':
        show_samples ^= 1;
        break;
      case 'R':
      case 'r':
        reset();
        if(show_samples) set_points();
        break;
      case 'S':
        num_samples = (num_samples-1)*2;
        break;
      case 's':
        num_samples = (num_samples/2)+1;
        break;
	  case '0':
		  set_meshes(0);
		  break;
	  case '1':
		  set_meshes(1);
		  break;
	  case '2':
		  set_meshes(2);
		  break;
#if 1
	  case 'i':
		  deform_init(VY, FY, VX, FX);
		  break;
	  case 'o':
	  {
		  Eigen::MatrixXd OY;
		  deform_solve(OY,
			  VY,
			  VX,
			  FX);
		  VY = OY;
		  set_meshes(2);
	  }
		  break;
#else
	  case 'i':
		  deform_init(VX, FX, VY, FY);
		  break;
	  case 'o':
	  {
		  Eigen::MatrixXd OY;
		  deform_solve(OY,
			  VX,
			  VY,
			  FY);
		  VX = OY;
		  set_meshes(1);
	  }
	  break;
#endif
	  default:
        return false;
    }
    return true;
  };

  reset();
  viewer.core.is_animating = false;
  viewer.core.point_size = 10;
  viewer.launch();
  return EXIT_SUCCESS;
}
