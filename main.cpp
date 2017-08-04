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
#include "optimal_nonrigid_icp.h"

Eigen::MatrixXd NFY;
Eigen::MatrixXd NVY;
Eigen::MatrixXi target_landmarks;
Eigen::MatrixXi template_landmarks;
const int nojaw = 66;

OptimalNonrigidICP nricp;

int main(int argc, char *argv[])
{
  srand(time(0));

  // Load input meshes
  Eigen::MatrixXd OVX,VX,VY,OVY;
  Eigen::MatrixXi FX,FY;
  igl::read_triangle_mesh(
	(argc>1 ? argv[1] : "../data/male.obj"), OVX, FX);
  igl::read_triangle_mesh(
    (argc>2 ? argv[2] : "../data/template.obj"), VY, FY);
  //VY.col(1) *= 0.75;
/*
  Eigen::MatrixXi I;
  Eigen::MatrixXd target_landmarks;
  igl::read_triangle_mesh(
	  (argc>3 ? argv[3] : "../data/male_feature.obj"), target_landmarks, I);
*/

  igl::per_face_normals(VY, FY, NFY);
  igl::per_vertex_normals(VY, FY, NVY);
  
#if 1
  // Find the bounding box and normalize data
  Eigen::Vector3d m = VY.colwise().minCoeff();
  Eigen::Vector3d M = VY.colwise().maxCoeff();
  Eigen::Vector3d cy = (m + M)*0.5f;
  const float scale = (M[0] - m[0]);

  m = OVX.colwise().minCoeff();
  M = OVX.colwise().maxCoeff();
  Eigen::Vector3d cx = (m + M)*0.5f;
  const float inv_scale = scale / (M[0] - m[0]);
  Eigen::RowVector3d offset(cy[0], cy[1], cy[2] + 1);
  //OVX.rowwise() -= cx.transpose();
  OVX *= inv_scale;
  std::cout << "scale: " << inv_scale << std::endl;
  //OVX.rowwise() += cy.transpose();
#elif 1
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
	igl::readDMAT("../data/male.dmat", target_landmarks);
	Eigen::MatrixXd target_landmark_points;
	target_landmark_points.resize(nojaw, 3);
	for (int i = 0, c = nojaw; i < c; ++i) {
		target_landmark_points.row(i) = OVX.row(target_landmarks.row(i )[0]);
	}

	igl::readDMAT("../data/head.dmat", template_landmarks);
	Eigen::MatrixXd template_landmark_points;
	template_landmark_points.resize(nojaw, 3);
	for (int i = 0, c = nojaw; i < c; ++i) {
		template_landmark_points.row(i) = VY.row(template_landmarks.row(i )[0]);
	}

	// align landmarks
	Eigen::Matrix3d R;
	Eigen::RowVector3d  t;
	for (int i = 0; i < 3; ++i)
	{
		point_to_point_rigid_matching(target_landmark_points, template_landmark_points, R, t);
		target_landmark_points = ((target_landmark_points*R).rowwise() + t).eval();
		OVX = ((OVX*R).rowwise() + t).eval();
	}
#else
	Eigen::MatrixXi I;
	Eigen::MatrixXd target_landmark_points;
	igl::read_triangle_mesh(
		(argc>3 ? argv[3] : "../data/male_feature.obj"), target_landmark_points, I);
	//target_landmark_points.rowwise() -= cx.transpose();
	target_landmark_points *= inv_scale;

	igl::readDMAT(
		(argc>4 ? argv[4] : "../data/template.dmat"), template_landmarks);
	Eigen::MatrixXd template_landmark_points;
	template_landmark_points.resize(template_landmarks.rows(), 3);
	for (int i = 0, c = template_landmarks.rows(); i < c; ++i) {
		template_landmark_points.row(i) = VY.row(template_landmarks.row(i)[0]);
	}

	// align landmarks
	Eigen::Matrix3d R;
	Eigen::RowVector3d  t;
	for (int i = 0; i < 1; ++i)
	{
		point_to_point_rigid_matching(target_landmark_points, template_landmark_points, R, t);
		target_landmark_points = ((target_landmark_points*R).rowwise() + t).eval();
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
  const Eigen::RowVector3d red(1.0, 0.1, 0.1);
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
#if 0
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
#else
	Eigen::MatrixXd XP(target_landmark_points.rows() + template_landmark_points.rows(), 3);
	XP << target_landmark_points, template_landmark_points;
	Eigen::MatrixXd C(XP.rows(), 3);
	C.array().topRows(target_landmark_points.rows()).rowwise() = (1. - (1. - blue.array())*.4);
	C.array().bottomRows(template_landmark_points.rows()).rowwise() = (1. - (1. - orange.array())*.8); 
	viewer.data.set_points(XP, C);
#endif
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

	  case 'q':
		  nricp.init(VX, FX, VY, FY);
		  break;
	  case 'w':
	  {
		  Eigen::MatrixXd OY;
		  nricp.compute(10, 1, OY);
		  VY = OY;
		  set_meshes(2);
	  }
		  break;
#if 1
	  case 'u':
		  deform_match(VY, FY, template_landmarks, VX, FX, target_landmarks);
		  set_meshes(2);
		  break;
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
