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
            int r = (int)((size_t)rand() % match_li.size()); 
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
    // std::cout << bestNormalizedHomography << std::endl;
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

void findDimensions(ezsift::Image<unsigned char> image, MatrixXd H, 
                    double *min_x, double *min_y, double *max_x, double *max_y){
    float h2 = image.h;
    float w2 = image.w;
    // creating endpoint coordinates (homogenous)
    /*np.float32([[0, 0,1], 
                    [0, w2,1], 
                    [h2, w2,1], 
                    [h2, 0,1]]).reshape(-1, 1, 2)
    */
    cv::Mat imgDimsTemp = cv::Mat::zeros (4, 3, CV_64F);
    imgDimsTemp.at<double>(1, 1) = h2; 
    imgDimsTemp.at<double>(2, 0) = w2; 
    imgDimsTemp.at<double>(2, 1) = h2; 
    imgDimsTemp.at<double>(3, 0) = w2; 
    imgDimsTemp.at<double>(0, 2) = 1.0; 
    imgDimsTemp.at<double>(1, 2) = 1.0; 
    imgDimsTemp.at<double>(2, 2) = 1.0; 
    imgDimsTemp.at<double>(3, 2) = 1.0; 
    
    // convert H into CV matrix 
    cv::Mat H_new ((int)(3), (int)(3), CV_64F);
    for (int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            H_new.at<double>(i, j) = H(i, j); 
        }
    }
    // Mapping to new end coordinates using H matrix
    cv::Mat imgDims = H_new * imgDimsTemp.t(); 
    cv::Mat lastRow = imgDims.row( 2 );
    cv::Mat tmp;
    cv::repeat(lastRow, 3, 1, tmp );
    // std::cout << "imgDims shape " << imgDims.rows << " " << imgDims.cols << std::endl;
    // std::cout << "last row shape " << lastRow.rows << " " << lastRow.cols << std::endl;
    imgDims = imgDims / tmp;
    // std::cout << "image dims " << imgDims << std::endl;

    // Finding min and max end points
    cv::minMaxLoc(imgDims.row(0), min_x, max_x, NULL, NULL);
    cv::minMaxLoc(imgDims.row(1), min_y, max_y, NULL, NULL);
    // printf("findDimensions: max y: %f, min_y: %f, max_x: %f, min_x: %f \n", *min_x, *min_y, *max_x, *max_y );
}

void placeImage(cv::Mat base, cv::Mat newImage){
    int w = newImage.cols;
    int h = newImage.rows;
    // printf("w: %d, h: %d", w, h);
    cv::Mat dstImg(base(cv::Rect(0, 0, w, h)));
    // cv::bitwise_or(dstImg, newImage, base(cv::Rect(0, 0, w, h))); 
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++){
            if (dstImg.at<uint8_t>(i, j) == 0){
                dstImg.at<uint8_t>(i, j) = newImage.at<uint8_t>(i, j);
            }
            if (dstImg.at<uint8_t>(i, j) != 0 && newImage.at<uint8_t>(i, j) != 0){
                dstImg.at<uint8_t>(i, j) = fmax(newImage.at<uint8_t>(i, j), dstImg.at<uint8_t>(i, j));
            }
        }
    }
}


int main(int argc, char *argv[])
{
    bool VIDEO_MODE;
    char* suffix = strrchr(argv[1], '.');
    if (argc < 2) {
        printf("Please input at least two image filenames or one video filename.\n");
        return -1;
    }

    if(strncmp(suffix, ".mp4", 4) == 0){
        VIDEO_MODE = true;  
    }else{
        VIDEO_MODE = false;
    }

    if (argc < 3 && !VIDEO_MODE) {
        printf("Please input at least two image filenames.\n");
        printf("usage: image_match img1 img2 ...\n");
        return -1;
    }

    
    std::vector<ezsift::Image<unsigned char> > images;
    std::vector<char * > files; //Should probably switch away from this when switching to video
    if(!VIDEO_MODE){
        // All image files
        for(int i=1; i<argc; i++){
            char* file = (char *)calloc(sizeof(char), strlen(argv[i]));
            memcpy(file, argv[i], sizeof(char) * strlen(argv[i]));
            file[strlen(argv[i])] = 0;
            files.push_back(file);
            ezsift::Image<unsigned char> image;

            //Finally can convert pngs
            cv::Mat pngImage = cv::imread(file, cv::IMREAD_UNCHANGED);
            cv::imwrite("tmp.pgm", pngImage);  

            if (image.read_pgm("tmp.pgm") != 0) {
                std::cerr << "Failed to open input image!" << std::endl;
                return -1;
            }
            images.push_back(image);
        }
    }else{
        char* file;
        memcpy(file, argv[1], sizeof(char) * strlen(argv[1]));
        file[strlen(argv[1])] = 0;
        // cv::VideoCapture cap(file);
        // if ( !cap.isOpened() ){  // isOpened() returns true if capturing has been initialized.
        //     std::cout << "Cannot open the video file. \n";
        //     return -1;
        // }
        // pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        // while(true){

        // }
    }
    
    
    std::vector<std::list<ezsift::MatchPair>> matches;
    ezsift::double_original_image(true);
    for(int i=0; i<images.size()-1; i++){
        int j = i+1;
        std::list<ezsift::SiftKeypoint> kpt_list1, kpt_list2;
        ezsift::sift_cpu(images[i], kpt_list1, true); //will write gpu version of this function
        ezsift::sift_cpu(images[j], kpt_list2, true);

        std::list<ezsift::MatchPair> match_list;
        ezsift::match_keypoints(kpt_list1, kpt_list2, match_list);
  
        matches.push_back(match_list);
        if(match_list.size() == 0){
            std::cerr << "Failed to find any matches between two adjacent images!" << std::endl;
            return -1;
        }
    }

    std::vector<MatrixXd> homographies;
    MatrixXd first = MatrixXd::Identity(3, 3);
    homographies.push_back(first);
    for(int i=1; i<images.size(); i++){
        MatrixXd bestH = computeRansac(matches[i-1]);
        homographies.push_back(homographies[i-1]*bestH);
    }

    int pano_min_x = 0; //std::numeric_limits<int>::max();
    int pano_min_y = 0; //std::numeric_limits<int>::max();
    int pano_max_x = images[0].w; //-std::numeric_limits<int>::max();
    int pano_max_y = images[0].h;  //-std::numeric_limits<int>::max();
    // printf("pano_max_x %d\n", pano_max_x);

    for(int i=1; i<images.size(); i++){
        double min_x;
        double min_y;
        double max_x;
        double max_y;
        findDimensions(images[i], homographies[i], &min_x, &min_y, &max_x, &max_y);

        pano_min_x = (int) fmax((floor(fmin(min_x, pano_min_x))),0); 
        pano_min_y = (int) fmax((floor(fmin(min_y, pano_min_y))),0); 
        pano_max_x = (int) (ceil(fmax(max_x, pano_max_x))); 
        pano_max_y = (int) (ceil(fmax(max_y, pano_max_y)));
        // printf("pano_size: max y: %d, min_y: %d, max_x: %d, min_x: %d \n", pano_max_y, pano_min_y, pano_max_x,pano_min_x );
    }

    int pan_height  = (int)(pano_max_y - pano_min_y); 
    int pan_width = (int)(pano_max_x - pano_min_x);

    cv::Mat resultImage (pan_height, pan_width, CV_8U);
    for (int i = 0; i < pan_height; i++){
        for (int j = 0; j < pan_width; j++){
            resultImage.at<uint8_t>(i, j) = 0;
        }
    }
    
    for (int i = 0; i < images.size(); i++){
        double min_x; 
        double min_y; 
        double max_x; 
        double max_y; 
        findDimensions(images[i], homographies[i], &min_x, &min_y, &max_x, &max_y); 
        // printf("max y: %lf, min_y: %lf, max_x: %lf, min_x: %lf \n", max_y, min_y, max_x, min_x );
        max_x = fmax(pano_max_x, max_x); 
        max_y = fmax(pano_max_y, max_y); 
        min_x = fmax(fmin(pano_min_x, min_x),0); 
        min_y = fmax(fmin(pano_min_y, min_y),0); 
        // printf("pano_size: max y: %d, min_y: %d, max_x: %d, min_x: %d \n", pano_max_y, pano_min_y, pano_max_x,pano_min_x );
        

        int curr_width = (int)(max_x - min_x);
        int curr_height  = (int)(max_y - min_y); 

        // convert homographies[i] into CV matrix 
        cv::Mat H (3, 3, CV_64F); 
        for (int r = 0; r < homographies[i].rows(); r++){
            for(int c = 0; c < homographies[i].cols(); c++){
                H.at<double>(r, c) = homographies[i](r, c); 
            }
        }
        // printf("after conversion H\n");

        cv::Mat newImg (curr_width, curr_height, CV_8U); 
        cv::Mat inpImg = cv::imread(files[i], cv::IMREAD_UNCHANGED); 
        // cv::Mat grayImage;
        // cv::cvtColor(inpImg, grayImage, cv::COLOR_BGR2GRAY);

        cv::warpPerspective(inpImg, newImg, H, cv::Size(curr_width, curr_height));

        // if (i == 0) cv::imwrite("temp1.png", newImg); 
        // if(i == 1) cv::imwrite("temp2.png", newImg); 
        // if(i == 2) cv::imwrite("temp3.png", newImg); 
        // if(i == 3) cv::imwrite("temp4.png", newImg); 
        // if(i == 4) cv::imwrite("temp5.png", newImg); 
        // if(i == 5) cv::imwrite("temp6.png", newImg); 

        placeImage(resultImage, newImg);
    }

    // Write out resultImg
    
    cv::imshow("windowName",resultImage);
    int k = cv::waitKey(0);
    if(k == 'q')
    {   
        cv::imwrite("result.png", resultImage); 
        return 0;
    }

    return 0;
}
