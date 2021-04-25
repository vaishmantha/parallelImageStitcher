#include "ezsift.h"

#include <iostream>
#include <list>
#include <eigen/Eigen/Dense>
 
using Eigen::MatrixXd;

#define USE_FIX_FILENAME 0

MatrixXd computeHomography(MatrixXd x1, MatrixXd x2){
    // printf("enter homograpjy\n"); 
    MatrixXd A;

    for (int i = 0; i < x1.rows(); i++){
        double x = x1(i, 0);
        double y = x1(i, 1);
        double x_p = x2(i, 0); 
        double y_p = x2(i, 1);

        if(i == 0){
            A = MatrixXd::Constant(2,9, 0.0);
            A << -x, -y, -1.0, 0.0, 0.0, 0.0, x*x_p, y*x_p, x_p,
                  0.0, 0.0, 0.0,-x, -y, -1.0, x*y_p, y*y_p, y_p;
        }else{
            MatrixXd B = MatrixXd::Constant(2,9, 0.0);
            B << -x, -y, -1.0, 0.0, 0.0, 0.0, x*x_p, y*x_p, x_p,
                  0.0, 0.0, 0.0,-x, -y, -1.0, x*y_p, y*y_p, y_p;
            
            MatrixXd C(A.rows()+B.rows(), A.cols());
            C << A, 
                B;
            A = C;
        }
    }
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeFullV);
    
    MatrixXd V = svd.matrixV();
    MatrixXd H = V.block(8,0,1,9);
    
    Eigen::Map<MatrixXd> finalH(H.data(), 3, 3); 
    return finalH.transpose();
}

MatrixXd computeNormalizedHomography(MatrixXd x1, MatrixXd x2, 
                                    MatrixXd homogeneous_x1, 
                                    MatrixXd homogeneous_x2){
    // x1 and x2 should be (1, 3)
    MatrixXd centroid1 = x1.colwise().mean(); //shape: (1,2)
    MatrixXd centroid2 = x2.colwise().mean(); //shape: (1,2)

    double scale1 = sqrt(2) / x1.colwise().norm().maxCoeff();
    double scale2 = sqrt(2) / x2.colwise().norm().maxCoeff();

    // similarity transform 1
    MatrixXd T1 = MatrixXd::Constant(3, 3, 0); 
    T1(0, 0) = scale1; 
    T1(1, 1) = scale1; 
    T1(2, 2) = 1; 

    T1(0, 2) = -1 * centroid1(0, 0) * scale1;
    T1(1, 2) = -1 * centroid1(0, 1) * scale1;

    // similarity transfor m2
    MatrixXd T2 = MatrixXd::Constant(3, 3, 0); 
    T2(0, 0) = scale2; 
    T2(1, 1) = scale2; 
    T2(2, 2) = 1; 

    T2(0, 2) = -1 * centroid2(0, 0) * scale2;
    T2(1, 2) = -1 * centroid2(0, 1) * scale2;

    // compute homography
    // printf("start4\n");
    MatrixXd x1Norm = (T1 * homogeneous_x1.transpose()).transpose();
    MatrixXd x2Norm = (T2 * homogeneous_x2.transpose()).transpose();
    // printf("end4\n");

    MatrixXd H = computeHomography(x1Norm(Eigen::all, {0, 1}), x2Norm(Eigen::all, {0, 1}));

    // denormalize
    return T1.inverse() * (H * T2); 
}

MatrixXd computeRansac(std::list<ezsift::MatchPair> match_li){
    int iterations= 1000; 
    int threshold = 250; //check on this threshold
    int maxCount = 0; 
    
    MatrixXd locs1 = MatrixXd(match_li.size(), 2);
    MatrixXd locs2 = MatrixXd(match_li.size(), 2);
    MatrixXd homogeneous_loc1 = MatrixXd(match_li.size(), 3);
    MatrixXd homogeneous_loc2 = MatrixXd(match_li.size(), 3);
    std::list<ezsift::MatchPair>::iterator itr;
    int i=0;
    for (itr = match_li.begin(); itr != match_li.end(); itr++){
        locs1(i, 0) = itr->r1;
        locs1(i, 1) = itr->c1;
        locs2(i, 0) = itr->r2;
        locs2(i, 1) = itr->c2;
        homogeneous_loc1(i, 0) = itr->r1;
        homogeneous_loc1(i, 1) = itr->c1;
        homogeneous_loc1(i, 2) = 1.0;
        homogeneous_loc2(i, 0) = itr->r2;
        homogeneous_loc2(i, 1) = itr->c2;
        homogeneous_loc2(i, 2) = 1.0;
        i++;
    }

    int max_count = 0;
    std::vector<int> inlier_inds; 
    int it; 
    for(it = 0; it < iterations; it++){
        std::vector<int> rand_inds; 
        while(rand_inds.size() != 4){
            double r = rand() % match_li.size(); 
            rand_inds.push_back(r);
        }
        for(auto v : rand_inds){
            std::cout << v << ", " << std::endl;
        }
        MatrixXd x1 = locs1(rand_inds, Eigen::all); //MatrixXd::Constant(rand_inds.size(), 3); 
        MatrixXd x2 = locs2(rand_inds, Eigen::all); //MatrixXd::Constant(rand_inds.size(), 3); 
        MatrixXd x1_res_h = homogeneous_loc1(rand_inds, Eigen::all); //MatrixXd::Constant(rand_inds.size(), 3); 
        MatrixXd x2_res_h = homogeneous_loc2(rand_inds, Eigen::all); //MatrixXd::Constant(rand_inds.size(), 3); 
       
        MatrixXd H = computeNormalizedHomography(x1, x2, x1_res_h, x2_res_h); 
        int count = 0; 
        MatrixXd prod = H * homogeneous_loc2.transpose();
        std::vector<int> inlier_inds_current; 
        double diff;
        //might need to turn back to non homogeneous coordinates below
        for(int i = 0; i < prod.rows(); i++){
            // printf("iterator: %d\n", i); 
            diff = (prod.transpose()(i, Eigen::all) - homogeneous_loc1(i, Eigen::all)).norm();
            if(diff < threshold){
                count++;
                inlier_inds_current.push_back(i);
            }
        }
        if (max_count <= count){
            max_count = count;
            inlier_inds = inlier_inds_current;
        }      
    }
    MatrixXd x1_res = locs1(inlier_inds, Eigen::all); 
    MatrixXd x2_res = locs2(inlier_inds, Eigen::all);
    MatrixXd x1_res_h = homogeneous_loc1(inlier_inds, Eigen::all); 
    MatrixXd x2_res_h = homogeneous_loc2(inlier_inds, Eigen::all);
    MatrixXd bestNormalizedHomography = computeNormalizedHomography(x1_res, x2_res, x1_res_h, x2_res_h);

    return bestNormalizedHomography;
}


int main(int argc, char *argv[])
{
#if USE_FIX_FILENAME
    char *file1 = "img1.pgm";
    char *file2 = "img2.pgm";
#else
    //TO DO: Add the ability to take a variable number of images
    if (argc != 3) {
        printf("Please input two image filenames.\n");
        printf("usage: image_match img1 img2\n");
        return -1;
    }
    char file1[255];
    char file2[255];
    memcpy(file1, argv[1], sizeof(char) * strlen(argv[1]));
    file1[strlen(argv[1])] = 0;
    memcpy(file2, argv[2], sizeof(char) * strlen(argv[2]));
    file2[strlen(argv[2])] = 0;
#endif

    // Read two input images
    ezsift::Image<unsigned char> image1, image2;
    if (image1.read_pgm(file1) != 0) {
        std::cerr << "Failed to open input image1!" << std::endl;
        return -1;
    }

    if (image2.read_pgm(file2) != 0) {
        printf("Failed to open input image2!\n");
        return -1;
    }

    // Double the original image as the first octive.
    ezsift::double_original_image(true); //CHECK WHAT THIS DOES

    // Detect keypoints
    std::list<ezsift::SiftKeypoint> kpt_list1, kpt_list2;
    ezsift::sift_cpu(image1, kpt_list1, true); //will write gpu version of this function

    ezsift::sift_cpu(image2, kpt_list2, true);

    // Save keypoint list, and draw keypoints on images.
    ezsift::draw_keypoints_to_ppm_file("sift_keypoints_a.ppm", image1,
                                       kpt_list1);

    ezsift::draw_keypoints_to_ppm_file("sift_keypoints_b.ppm", image2,
                                       kpt_list2);

    // Match keypoints.
    std::list<ezsift::MatchPair> match_list;
    /*
    struct MatchPair {
        int r1;
        int c1;
        int r2;
        int c2;
    };
    */
    ezsift::match_keypoints(kpt_list1, kpt_list2, match_list);

    // Draw result image where the two images are matched
    ezsift::draw_match_lines_to_ppm_file("sift_matching_a_b.ppm", image1,
                                         image2, match_list);
    
    auto bestH = computeRansac(match_list);
    std::cout << "homography: " << bestH << std::endl;
    return 0;
}
