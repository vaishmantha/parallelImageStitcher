#include "ezsift.h"

#include <iostream>
#include <list>
#include <eigen/Eigen/Dense>
 
using Eigen::MatrixXd;

#define USE_FIX_FILENAME 0

// MatrixXd computeHomography(std::list<ezsift::MatchPair> match_list){
//     MatrixXd A;

//     std::list<ezsift::MatchPair>::iterator it;
//     for (it = match_list.begin(); it != match_list.end(); it++){
//         int y = it->r1;
//         int x = it->c1;
//         int y_p = it->r2; 
//         int x_p = it->c2;

//         if(it == match_list.begin()){
//             A = MatrixXd::Constant(2,9, 0);
//             A << -x, -y, -1, 0, 0, 0, x*x_p, y*x_p, x_p,
//                   0, 0, 0,-x, -y, -1, x*y_p, y*y_p, y_p;
//         }else{
//             MatrixXd B = MatrixXd::Constant(2,9, 0);
//             B << -x, -y, -1, 0, 0, 0, x*x_p, y*x_p, x_p,
//                   0, 0, 0,-x, -y, -1, x*y_p, y*y_p, y_p;

//             MatrixXd C(A.rows()+B.rows(), A.cols());
//             C << A, 
//                 B;
//             A = C;
//         }
//     }
//     Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeFullV);
    
//     MatrixXd V = svd.matrixV();
//     std::cout << V << std::endl;

//     MatrixXd H = V.block(8,0,1,9);
    
//     Eigen::Map<MatrixXd> finalH(H.data(), 3, 3); 
//     std::cout << finalH.transpose()<< std::endl;
//     return finalH.transpose();
// }

MatrixXd computeHomography(MatrixXd x1, MatrixXd x2){
    MatrixXd A;

    for (int i = 0; i < x1.rows(); i++){
        // QUESTION: what is this x y switching 
        int y = x1(i, 0);
        int x = x1(i, 1);
        int y_p = x2(i, 0); 
        int x_p = x2(i, 1);

        if(i == 0){
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

MatrixXd computeNormalizedHomography(MatrixXd x1, MatrixXd x2){
    // x1 and x2 should be (1, 3)

    MatrixXd centroid1 = x1.colwise().mean();
    MatrixXd centroid2 = x2.colwise().mean();

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
    MatrixXd x1Norm = (T1 * x1.transpose()).transpose();
    MatrixXd x2Norm = (T2 * x2.transpose()).transpose();

    MatrixXd H = computeHomography(x1Norm(Eigen::all, {0, 1}), x2Norm(Eigen::all, {0, 1}));

    // denormalize
    return T1.inverse() * (H * T2); 
}

MatrixXd computeRansac(std::list<ezsift::MatchPair> match_li){
    std::vector<ezsift::MatchPair> match_list(match_li.begin(), match_li.end());
    int iterations= 1000; 
    int threshold = 57; 
    int maxCount = 0; 
    std::list<bool> inliers; 
    int it; 
    for(it = 0; it < iterations; it++){
        std::vector<int> rand_inds; 
        while(rand_inds.size() != 4){
            // FIX: randomization might be sus syntax; need to make sure randoms don't repeat but not sure if C++ takes care of that 
            double r = rand() % match_list.size(); 
            rand_inds.push_back(r);
        }
        MatrixXd x1 = MatrixXd::Constant(rand_inds.size(), 3); 
        MatrixXd x2 = MatrixXd::Constant(rand_inds.size(), 3); 
        // rand_inds.size() should be 4
        for (int i  = 0; i < rand_inds.size(); i++){
            // FIX: check r1 c1 r2 c2 order 
            x1(i, 0) = match_list.at(rand_inds.at(i)).r1;
            x1(i, 1) = match_list.at(rand_inds.at(i)).c1;
            x2(i, 0) = match_list.at(rand_inds.at(i)).r2;
            x2(i, 1) = match_list.at(rand_inds.at(i)).c2;
            // homogenous coordinates
            x1(i, 2) = 1; 
            x2(i, 2) = 1; 
        }
        MatrixXd H = computeNormalizedHomography(x1, x2); 
        int count = 0; 
        std::list<bool> inlier_current; 
        for(int i = 0; i < match_list.size(); i++){
            // INCOMPLETE BUT NEED TO RESUME BELOW PYTHON CODE
            // MatrixXd l2 = H * 
        }
    }


    // def computeH_ransac(locs1, locs2):
    // iterations = 10000
    // # threshold = 50
    // threshold = 57
    // max_count = 0
    // inliers = list()
    // for it in range(iterations):
    //     rand_inds = list()
    //     while len(rand_inds) != 4:
    //         r = np.random.randint(0, len(locs1))
    //         if r not in rand_inds: rand_inds.append(r)
    //     x1 = list()
    //     x2 = list()
    //     for i in rand_inds:
    //         x1.append(locs1[i])
    //         x2.append(locs2[i])
    //     H = computeH_norm(x1,x2)
    //     count = 0
    //     inlier_current = list()
    //     for x in range(len(locs1)):
    //         l2 = np.matmul(H, (np.append(locs2[x],1)))
    //         l1 = (np.append(locs1[x],1))
    //         diff = np.linalg.norm(np.subtract(l2, l1))
    //         if diff < threshold:
    //             count += 1
    //             inlier_current.append(True)
    //         else:
    //             inlier_current.append(False)
    //     if max_count <= count:
    //         max_count = count
    //         inliers = inlier_current
    // x1_res = list()
    // x2_res = list()
    // for x in range(len(inliers)):
    //     if inliers[x]:
    //         x1_res.append(locs1[x])
    //         x2_res.append(locs2[x])
    // bestH2to1 = computeH_norm(x1_res, x2_res)
    // return bestH2to1, np.array(inliers)
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
    
    computeNormalizedHomography(match_list);

    return 0;
}
