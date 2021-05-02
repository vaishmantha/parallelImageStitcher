#include "ezsift.h"
#include <limits>
#include <iostream>
#include <list>
#include <eigen/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// #undef cimg_display
// #define cimg_display 0
// #include "CImg.h"

// using namespace cimg_library;

using Eigen::MatrixXd;

#define USE_FIX_FILENAME 0

MatrixXd computeHomography(MatrixXd x1, MatrixXd x2){
    MatrixXd A;

    for (int i = 0; i < x1.rows(); i++){
        // QUESTION: what is this x y switching 
        // printf("%d \n", i);
        double x = x2(i, 0);
        double y = x2(i, 1);
        double x_p = x1(i, 0); 
        double y_p = x1(i, 1);

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
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeFullV);
    
    MatrixXd V = svd.matrixV();
    // eigen vector corresponding to smallest singular value
    MatrixXd H = V.block(0,8,9,1); //make sure this is the right one
    
    Eigen::Map<MatrixXd> finalH(H.data(), 3, 3); 

    return finalH.transpose();
}

MatrixXd computeNormalizedHomography(MatrixXd x1, MatrixXd x2, 
                                    MatrixXd homogeneous_x1, 
                                    MatrixXd homogeneous_x2){
    // x1 and x2 should be (1, 3)
    
    MatrixXd centroid1 = x1.colwise().mean(); //shape: (1,2)
    MatrixXd centroid2 = x2.colwise().mean(); //shape: (1,2)
    // std::cout << "Centroid 1 " << centroid1 << " \n" << "Centroid 2 " << centroid2 << std::endl;

    double scale1 = sqrt(2) / x1.colwise().norm().maxCoeff();
    double scale2 = sqrt(2) / x2.colwise().norm().maxCoeff();

    // similarity transform 1
    MatrixXd T1 = MatrixXd::Constant(3, 3, 0.0); //correct
    T1(0, 0) = scale1; 
    T1(1, 1) = scale1; 
    T1(2, 2) = 1.0; 

    T1(0, 2) = -1.0 * centroid1(0, 0) * scale1;
    T1(1, 2) = -1.0 * centroid1(0, 1) * scale1;

    // similarity transfor m2
    MatrixXd T2 = MatrixXd::Constant(3, 3, 0.0); //correct
    
    T2(0, 0) = scale2; 
    T2(1, 1) = scale2; 
    T2(2, 2) = 1.0; 

    T2(0, 2) = -1.0 * centroid2(0, 0) * scale2;
    T2(1, 2) = -1.0 * centroid2(0, 1) * scale2;

    // compute homography
    MatrixXd x1Norm = (T1 * homogeneous_x1.transpose()).transpose();
    MatrixXd x2Norm = (T2 * homogeneous_x2.transpose()).transpose();

    MatrixXd H = computeHomography(x1Norm(Eigen::all, {0, 1}), x2Norm(Eigen::all, {0, 1}));
    // std::cout << "H before normalization " << H << std::endl;

    // std::cout << "H after normalization " << T1.inverse() * (H * T2) << std::endl;
    // denormalize
    // printf("end norm\n"); 
    return T1.inverse() * (H * T2);  //correct
}


MatrixXd computeRansac(std::list<ezsift::MatchPair> match_li){
    // printf("start ransac\n"); 
    int iterations= 1000; 
    int threshold = 3; //check on this threshold
    int maxCount = 0; 
    
    MatrixXd locs1 = MatrixXd(match_li.size(), 2);
    MatrixXd locs2 = MatrixXd(match_li.size(), 2);
    MatrixXd homogeneous_loc1 = MatrixXd(match_li.size(), 3);
    MatrixXd homogeneous_loc2 = MatrixXd(match_li.size(), 3);
    std::list<ezsift::MatchPair>::iterator itr;
    int i=0;
    for (itr = match_li.begin(); itr != match_li.end(); itr++){
        locs1(i, 0) = itr->c1;
        locs1(i, 1) = itr->r1;
        locs2(i, 0) = itr->c2;
        locs2(i, 1) = itr->r2;
        homogeneous_loc1(i, 0) = itr->c1;
        homogeneous_loc1(i, 1) = itr->r1;
        homogeneous_loc1(i, 2) = 1.0;
        homogeneous_loc2(i, 0) = itr->c2;
        homogeneous_loc2(i, 1) = itr->r2;
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
        MatrixXd x1 = locs1(rand_inds, Eigen::all); 
        MatrixXd x2 = locs2(rand_inds, Eigen::all); 

        MatrixXd x1_res_h = homogeneous_loc1(rand_inds, Eigen::all); 
        MatrixXd x2_res_h = homogeneous_loc2(rand_inds, Eigen::all); 

        MatrixXd H = computeNormalizedHomography(x1, x2, x1_res_h, x2_res_h); 
        int count = 0; 
        MatrixXd prod = H * homogeneous_loc2.transpose();
        std::vector<int> inlier_inds_current; 
        double diff;
        bool divide_by_zero = false;

        for(int i = 0; i < prod.cols(); i++){
            if(prod.transpose()(i, 2) == 0){
                divide_by_zero = true;
                break;
            }
            diff = (prod.transpose()(i, {0,1})/prod.transpose()(i, 2) - locs1(i, Eigen::all)).norm();
            if(diff < threshold){
                count++;
                inlier_inds_current.push_back(i);
            }
        }
        if (!divide_by_zero && max_count <= count){
            max_count = count;
            inlier_inds = inlier_inds_current;
        }      
    }
    MatrixXd x1_res = locs1(inlier_inds, Eigen::all); 
    MatrixXd x2_res = locs2(inlier_inds, Eigen::all);
    MatrixXd x1_res_h = homogeneous_loc1(inlier_inds, Eigen::all); 
    MatrixXd x2_res_h = homogeneous_loc2(inlier_inds, Eigen::all);
    MatrixXd bestNormalizedHomography = computeNormalizedHomography(x1_res, x2_res, x1_res_h, x2_res_h);
    std::cout << bestNormalizedHomography << std::endl;
    return bestNormalizedHomography;
}

cv::Mat imageStitch(MatrixXd bestNormalizedH, char* file1, char* file2){
    cv::Mat H ((int)(bestNormalizedH.rows()), (int)(bestNormalizedH.cols()), CV_64F);
    for (int i = 0; i < bestNormalizedH.rows(); i++){
        for(int j = 0; j < bestNormalizedH.cols(); j++){
            H.at<double>(i, j) = bestNormalizedH(i, j); 
        }
    } 
    cv::Mat temp = cv::imread(file1); 
    cv::Mat img = cv::imread(file2); 

    int width = temp.size().width + img.size().width; 
    int height = temp.size().height + img.size().height;
    cv::Mat dstImg2; 

    cv::warpPerspective(img, dstImg2, H, cv::Size(width, height)); 

    cv::Mat dstImg(dstImg2(cv::Rect(0, 0, temp.size().width, temp.size().height)));
    temp.copyTo(dstImg);

    return dstImg2; 
}

void findDimensions(std::list<ezsift::MatchPair> matches, MatrixXd H, 
                    int *min_x, int *min_y, int *max_x, int *max_y){
    
}


int main(int argc, char *argv[])
{

    //TO DO: Add the ability to take a variable number of images
    if (argc < 3) {
        printf("Please input at least two image filenames.\n");
        printf("usage: image_match img1 img2 ...\n");
        return -1;
    }

    std::vector<ezsift::Image<unsigned char> > images;
    std::vector<char * > files;
    // All image files
    for(int i=1; i<argc; i++){
        char file[255];
        memcpy(file, argv[i], sizeof(char) * strlen(argv[i]));
        file[strlen(argv[i])] = 0;
        files.push_back(file);
        ezsift::Image<unsigned char> image;

        //Finally can convert pngs
        cv::Mat pngImage = cv::imread(file);
        cv::Mat grayImage;
        cv::cvtColor(pngImage, grayImage, cv::COLOR_BGR2GRAY);
        cv::imwrite("tmp.pgm", grayImage);  

        if (image.read_pgm("tmp.pgm") != 0) {
            std::cerr << "Failed to open input image!" << std::endl;
            return -1;
        }
        images.push_back(image);
    }
    
    std::vector<std::list<ezsift::MatchPair>> matches;
    for(int i=0; i<images.size()-1; i++){
        int j = i+1;
        std::list<ezsift::SiftKeypoint> kpt_list1, kpt_list2;
        // ezsift::double_original_image(true);
        ezsift::sift_cpu(images[i], kpt_list1, true); //will write gpu version of this function
        ezsift::sift_cpu(images[j], kpt_list2, true);

        std::list<ezsift::MatchPair> match_list;
        ezsift::match_keypoints(kpt_list1, kpt_list2, match_list);
  
        matches.push_back(match_list);
    }

    std::vector<MatrixXd> homographies;
    for(int i=0; i<images.size()-1; i++){
        MatrixXd bestH = computeRansac(matches[i]);
        homographies.push_back(bestH);
    }

    MatrixXd accH = MatrixXd::Identity(3,3); 

    printf("Now combining the Hs\n");
    int pano_min_x = std::numeric_limits<int>::max();
    int pano_min_y = std::numeric_limits<int>::max();
    int pano_max_x = std::numeric_limits<int>::min();
    int pano_max_y = std::numeric_limits<int>::min();
    for(int i=0; i<images.size()-1; i++){
        int min_x;
        int min_y;
        int max_x;
        int max_y;
        findDimensions(matches[i], homographies[i], &min_x, &min_y, &max_x, &max_y);
    }
    //Needs to be sequential
    // cv::Mat resultImage;
    // for(int i=images.size()-1; i>0;i--){
    //     printf("Case 1 before before\n");
    //     accH = homographies[i-1];
    //     char res[255];
    //     strcpy(res, "result.png");
    //     if(i == images.size()-1){
    //         printf("Case 1 before\n");
    //         cv::Mat resultImage = imageStitch(accH, files[i-1], files[i]); //1,2
    //         cv::imwrite("result.png", resultImage); 
    //         printf("Case 1 after\n");
    //     }else{
    //         printf("Case 2 before\n");
    //         cv::Mat resultImage = imageStitch(accH, files[i-1], res);
    //         printf("Case 2 after\n");
    //         cv::imwrite("result.png", resultImage); 
    //     }
    // }
    // cv::Mat resultImage;
    // for(int i=0; i<images.size()-1;i++){
    //     printf("Case 1 before before\n");
    //     accH = accH*homographies[i];
    //     char res[255];
    //     strcpy(res, "result.png");
    //     if(i == 0){
    //         printf("Case 1 before\n");
    //         cv::Mat resultImage = imageStitch(accH, files[i], files[i+1]); //1,2
    //         cv::imwrite("result.png", resultImage); 
    //         printf("Case 1 after\n");
    //     }else{
    //         printf("Case 2 before\n");
    //         cv::Mat resultImage = imageStitch(accH, res, files[i+1]);
    //         printf("Case 2 after\n");
    //         cv::imwrite("result.png", resultImage); 
    //     }
    // }
    
    
    //////////////////

    // auto bestH = computeRansac(match_list);
    // auto resultImage = imageStitch(bestH, file1, file2);
    // cv::imwrite("result.png", resultImage); 

    // char file1[255];
    // char file2[255];
    // printf("%s", argv[1]);
    // memcpy(file1, argv[1], sizeof(char) * strlen(argv[1]));
    // file1[strlen(argv[1])] = 0;
    // memcpy(file2, argv[2], sizeof(char) * strlen(argv[2]));
    // file2[strlen(argv[2])] = 0;
// #endif
    
    // Read two input images
    // ezsift::Image<unsigned char> image1, image2;
    // if (image1.read_pgm(file1) != 0) {
    //     std::cerr << "Failed to open input image1!" << std::endl;
    //     return -1;
    // }

    // if (image2.read_pgm(file2) != 0) {
    //     printf("Failed to open input image2!\n");
    //     return -1;
    // }

    // Double the original image as the first octive.
    // ezsift::double_original_image(true); //CHECK WHAT THIS DOES

    // Detect keypoints
    // std::list<ezsift::SiftKeypoint> kpt_list1, kpt_list2;
    // ezsift::sift_cpu(image1, kpt_list1, true); //will write gpu version of this function

    // ezsift::sift_cpu(image2, kpt_list2, true);

    // Save keypoint list, and draw keypoints on images.
    // ezsift::draw_keypoints_to_ppm_file("sift_keypoints_a.ppm", image1,
    //                                    kpt_list1);

    // ezsift::draw_keypoints_to_ppm_file("sift_keypoints_b.ppm", image2,
    //                                    kpt_list2);

    // Match keypoints.
    // std::list<ezsift::MatchPair> match_list;
    // ezsift::match_keypoints(kpt_list1, kpt_list2, match_list);

    // Draw result image where the two images are matched
    // ezsift::draw_match_lines_to_ppm_file("sift_matching_a_b.ppm", image1,
    //                                      image2, match_list);
    
    // auto bestH = computeRansac(match_list);
    // auto resultImage = imageStitch(bestH, file1, file2);
    // // std::cout << resultImage << std::endl;
    // cv::imwrite("result.png", resultImage);  
    return 0;
}
