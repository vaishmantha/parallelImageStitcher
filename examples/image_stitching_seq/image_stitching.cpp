#include "ezsift.h"

#include <iostream>
#include <list>
#include <eigen/Eigen/Dense>
 
using Eigen::MatrixXd;

#define USE_FIX_FILENAME 0

MatrixXd computeHomography(std::list<ezsift::MatchPair> match_list){
    MatrixXd A;

    std::list<ezsift::MatchPair>::iterator it;
    for (it = match_list.begin(); it != match_list.end(); it++){
        int y = it->r1;
        int x = it->c1;
        int y_p = it->r2; 
        int x_p = it->c2;

        if(it == match_list.begin()){
            A = MatrixXd::Constant(2,9, 0);
            A << -x, -y, -1, 0, 0, 0, x*x_p, y*x_p, x_p,
                  0, 0, 0,-x, -y, -1, x*y_p, y*y_p, y_p;
        }else{
            MatrixXd B = MatrixXd::Constant(2,9, 0);
            B << -x, -y, -1, 0, 0, 0, x*x_p, y*x_p, x_p,
                  0, 0, 0,-x, -y, -1, x*y_p, y*y_p, y_p;

            MatrixXd C(A.rows()+B.rows(), A.cols());
            C << A, 
                B;
            A = C;
        }
    }
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeFullV);
    
    MatrixXd V = svd.matrixV();
    std::cout << V << std::endl;

    MatrixXd H = V.block(8,0,1,9);
    
    Eigen::Map<MatrixXd> finalH(H.data(), 3, 3); 
    std::cout << finalH.transpose()<< std::endl;
    return finalH.transpose();
}

MatrixXd computeNormalizedHomography(){
    std::list<ezsift::MatchPair> match_list;
    
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
    
    computeHomography(match_list);

    return 0;
}
