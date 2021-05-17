#include "ezsift.h"
#include "CycleTimer.h"
#include <limits>
#include <iostream>
#include <list>
#include <eigen/Eigen/Dense>

#include "lodepng/lodepng.h"

using Eigen::MatrixXd;

#define USE_FIX_FILENAME 0

void dummyWarmup();
void ransacIterationDiff(MatrixXd prod, MatrixXd locs1, int threshold, int* count);
// void placeImage(MatrixXd newImage, MatrixXd* resImg, double min_x, double min_y, double max_x, double max_y);
void warpPerspective(unsigned char* png_r, unsigned char* png_g, unsigned char* png_b, 
        int png_width, int png_height, unsigned char* newImR, unsigned char* newImG, unsigned char* newImB, 
        MatrixXd H, int curr_width, int curr_height);

MatrixXd Matslice(MatrixXd array, int start_row, int start_col, int height, int width){
    MatrixXd sl = MatrixXd::Constant(height, width, 0);
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            sl(i, j) = array(start_row+i, start_col+j);
        }
    }
    return sl;
}

MatrixXd MatVectorslice(MatrixXd array, std::vector<int> row_indices, int start_col, int width){
    MatrixXd sl = MatrixXd::Constant(row_indices.size(), width, 0);
    for(int i=0; i<row_indices.size(); i++){
        for(int j=0; j<width; j++){
            sl(i, j) = array(row_indices[i], start_col+j);
        }
    }
    return sl;
}


MatrixXd MatVectorslice2(MatrixXd array, int* row_indices, int num_row_indices, int start_col, int width){
    MatrixXd sl = MatrixXd::Constant(num_row_indices, width, 0);
    for(int i=0; i<num_row_indices; i++){
        for(int j=0; j<width; j++){
            sl(i, j) = array(row_indices[i], start_col+j);
        }
    }
    return sl;
}

MatrixXd computeHomography(MatrixXd x1, MatrixXd x2){
    MatrixXd A;

    for (int i = 0; i < x1.rows(); i++){
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

// void warpPerspective(unsigned char* png_r, unsigned char* png_g, unsigned char* png_b, unsigned char* png_a, 
//         int png_width, int png_height, unsigned char* newImR, unsigned char* newImG, unsigned char* newImB, unsigned char* newImA, 
//         MatrixXd H, int curr_width, int curr_height){
//     int i; 
//     for(i=0; i< png_height; i++){ 
//         for(int j=0; j<png_width; j++){
//             MatrixXd tmp = MatrixXd::Constant(3,1, 0.0);
//             tmp(0,0) = j;
//             tmp(1,0) = i;
//             tmp(2,0) = 1;
//             MatrixXd res = H*tmp;
//             MatrixXd tm =  Matslice(res, 2, 0, 1, res.cols()).replicate(3,1); //(MatrixXd array, int start_row, int start_col, int height, int width)
//             res = res.cwiseQuotient(tm);
//             if ((int)res(0,0) >= 0 && (int)res(0,0) < curr_width && (int)res(1,0) >= 0 && (int)res(1,0) < curr_height){
//                 newImR[((int)res(1,0))*curr_width +  (int)res(0,0)] = (int)png_r[i*png_width + j];
//                 newImG[((int)res(1,0))*curr_width +  (int)res(0,0)] = (int)png_g[i*png_width + j];
//                 newImB[((int)res(1,0))*curr_width +  (int)res(0,0)] = (int)png_b[i*png_width + j];
//                 newImA[((int)res(1,0))*curr_width +  (int)res(0,0)] = (int)png_a[i*png_width + j];
//             }
//         }
//     }
// }

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

    MatrixXd H = computeHomography(Matslice(x1Norm, 0, 0, x1Norm.rows(), 2), Matslice(x2Norm, 0, 0, x2Norm.rows(), 2));
    return T1.inverse() * (H * T2);  //correct
}

    


MatrixXd computeRansac(std::list<ezsift::MatchPair> match_li){
    int iterations= 1000; 
    int threshold = 3; //check on this threshold
    
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


    int* rand_inds = (int*) calloc(sizeof(int), 4 * iterations);
    // #pragma omp parallel for schedule(dynamic) // DO NOT ADD BACK
    for(int it = 0; it < iterations; it++){
        int rand_counter = 0; 
        while(rand_counter != 4){
            int r = (int)((size_t)rand() % match_li.size()); 
            rand_inds[4*it + rand_counter] = r;
            rand_counter++; 
        }
    }

    int *count_list = (int*)calloc(iterations, sizeof(int));
    int it; 
    double loopStart = CycleTimer::currentSeconds();
    #pragma omp parallel for schedule(dynamic)
    for(it = 0; it < iterations; it++){
        MatrixXd x1 = MatVectorslice2(locs1, &rand_inds[4 * it], 4, 0, locs1.cols()); //locs1(rand_inds, Eigen::seqN(0,locs1.cols())); 
        MatrixXd x2 = MatVectorslice2(locs2, &rand_inds[4 * it], 4, 0, locs2.cols());// locs2(rand_inds, Eigen::seqN(0,locs2.cols())); 

        MatrixXd x1_res_h = MatVectorslice2(homogeneous_loc1, &rand_inds[4 * it], 4, 0, homogeneous_loc1.cols()); //homogeneous_loc1(rand_inds,  Eigen::seqN(0,homogeneous_loc1.cols())); 
        MatrixXd x2_res_h = MatVectorslice2(homogeneous_loc2, &rand_inds[4 * it], 4, 0, homogeneous_loc2.cols()); //homogeneous_loc2(rand_inds, Eigen::seqN(0,homogeneous_loc2.cols())); 

        MatrixXd H = computeNormalizedHomography(x1, x2, x1_res_h, x2_res_h); 
        int count = 0; 
        // double prodStart = CycleTimer::currentSeconds();
        MatrixXd prod = H * homogeneous_loc2.transpose();
        double diff;
        bool divide_by_zero = false;
        
        // double insideLoopStart = CycleTimer::currentSeconds();
        //ransacIterationDiff(prod, locs1, threshold, &count);
        for(int i = 0; i < prod.cols(); i++){ //FIX: 100s of cols here, so cudify
            if(prod.transpose()(i, 2) == 0){
                divide_by_zero = true;
            }
            if(!divide_by_zero){
                // diff = sqrt(pow((prod(0,i)/prod(2, i) - locs1(i,0)),2) + pow((prod(1,i)/prod(2,i) - locs1(i,1)), 2)); 
                diff = (Matslice(prod.transpose(), i, 0, 1, 2)/prod.transpose()(i, 2) - Matslice(locs1, i, 0, 1, locs1.cols())).norm(); 
                if(diff < threshold){
                    count++;
                }
            }
        }
        
        // double insideLoopEnd = CycleTimer::currentSeconds();
        // std::cout << "Inside loop time: " << insideLoopEnd - insideLoopStart << std::endl;

        if (!divide_by_zero){
            count_list[it] = count;
        }      
    }

    // double loopEnd = CycleTimer::currentSeconds();
    // std::cout << "Loop ransac time" << loopEnd - loopStart << std::endl;

    int max_count = -1;
    int k; 
    int best_i; 
    #pragma omp parallel for reduction(max: max_count)
    for(k = 0; k < iterations; k++){
        if(max_count < count_list[k]){
            max_count = count_list[k];
            best_i = k; 
        }
    }

    // double Start = CycleTimer::currentSeconds();
    // found the best one 
    MatrixXd x1 = MatVectorslice2(locs1, &rand_inds[4 * best_i], 4, 0, locs1.cols()); //locs1(rand_inds, Eigen::seqN(0,locs1.cols())); 
    MatrixXd x2 = MatVectorslice2(locs2, &rand_inds[4 * best_i], 4, 0, locs2.cols());// locs2(rand_inds, Eigen::seqN(0,locs2.cols())); 

    MatrixXd x1_res_h = MatVectorslice2(homogeneous_loc1, &rand_inds[4 * best_i], 4, 0, homogeneous_loc1.cols()); //homogeneous_loc1(rand_inds,  Eigen::seqN(0,homogeneous_loc1.cols())); 
    MatrixXd x2_res_h = MatVectorslice2(homogeneous_loc2, &rand_inds[4 * best_i], 4, 0, homogeneous_loc2.cols()); //homogeneous_loc2(rand_inds, Eigen::seqN(0,homogeneous_loc2.cols())); 

    MatrixXd H = computeNormalizedHomography(x1, x2, x1_res_h, x2_res_h); 
    MatrixXd prod = H * homogeneous_loc2.transpose();
    std::vector<int> inlier_inds; 
    double diff;
    bool divide_by_zero = false;
    for(int i = 0; i < prod.cols(); i++){
        if(prod.transpose()(i, 2) == 0){
            divide_by_zero = true;
        }
        if(!divide_by_zero){
            diff = (Matslice(prod.transpose(), i, 0, 1, 2)/prod.transpose()(i, 2) - Matslice(locs1, i, 0, 1, locs1.cols())).norm(); 
            // diff = (prod.transpose()(i, {0,1})/prod.transpose()(i, 2) - Matslice(locs1, i, 0, 1, locs1.cols())).norm(); 
            if(diff < threshold){
                inlier_inds.push_back(i);
            }
        }
    }
    // double End = CycleTimer::currentSeconds();
    // std::cout << "End of ransac time" << End - Start << std::endl;

    // double actStart = CycleTimer::currentSeconds();
    MatrixXd x1_res = MatVectorslice(locs1, inlier_inds, 0, locs1.cols()); //locs1(inlier_inds, Eigen::seqN(0,locs1.cols())); 
    MatrixXd x2_res = MatVectorslice(locs2, inlier_inds, 0, locs2.cols()); //locs2(inlier_inds, Eigen::seqN(0,locs2.cols()));
    MatrixXd x1_res_h2 = MatVectorslice(homogeneous_loc1, inlier_inds, 0, homogeneous_loc1.cols()); //homogeneous_loc1(inlier_inds, Eigen::seqN(0,homogeneous_loc1.cols())); 
    MatrixXd x2_res_h2 =  MatVectorslice(homogeneous_loc2, inlier_inds, 0, homogeneous_loc2.cols()); //homogeneous_loc2(inlier_inds, Eigen::seqN(0,homogeneous_loc2.cols()));
    MatrixXd bestNormalizedHomography = computeNormalizedHomography(x1_res, x2_res, x1_res_h2, x2_res_h2);
    // double actEnd = CycleTimer::currentSeconds();
    // std::cout << "Actual End of ransac time" << actEnd - actStart << std::endl;
    return bestNormalizedHomography;
}

void findDimensions(ezsift::Image<unsigned char> image, MatrixXd H, 
                    double *min_x, double *min_y, double *max_x, double *max_y){
    float h2 = image.h;
    float w2 = image.w;

    MatrixXd imgDimsTmp = MatrixXd::Constant(4,3,0.0);
    imgDimsTmp(1,1) = h2;
    imgDimsTmp(2,0) = w2;
    imgDimsTmp(2,1) = h2;
    imgDimsTmp(3,0) = w2;
    imgDimsTmp(0,2) = 1.0;
    imgDimsTmp(1,2) = 1.0;
    imgDimsTmp(2,2) = 1.0;
    imgDimsTmp(3,2) = 1.0;

    MatrixXd imgDimss = H*(imgDimsTmp.transpose());
    MatrixXd lRow = Matslice(imgDimss, 2, 0, 1, imgDimss.cols()); 
    MatrixXd tm = lRow.replicate(3,1);
    for(int i=0; i< tm.rows(); i++){
        for(int j=0; j< tm.cols(); j++){
            tm(i,j) = 1.0/tm(i,j);
        }
    }
    imgDimss = imgDimss.cwiseProduct(tm);
    *min_x = Matslice(imgDimss, 0, 0, 1, imgDimss.cols()).minCoeff(); 
    *max_x = Matslice(imgDimss, 0, 0, 1, imgDimss.cols()).maxCoeff();
    *min_y = Matslice(imgDimss, 1, 0, 1, imgDimss.cols()).minCoeff(); 
    *max_y = Matslice(imgDimss, 1, 0, 1, imgDimss.cols()).maxCoeff(); 
}

// void placeImage(unsigned char* newImage, int newImWidth, int newImHeight, MatrixXd* resImg, double min_x, double min_y, double max_x, double max_y){
//     int w = newImWidth; //newImage.cols();
//     int h = newImHeight; //newImage.rows();
//     // printf("w: %d, h: %d", w, h);
//     for (int i = 0; i < h; i++){ //access as row col
//         for (int j = 0; j < w; j++){
//             if ((*resImg)(i,j) == 0){
//                 (*resImg)(i,j) = newImage[i*newImWidth + j]; //(i,j);
//             }
//             if ((*resImg)(i,j) != 0 && newImage[i*newImWidth + j] != 0){
//                 (*resImg)(i,j) = fmax(newImage[i*newImWidth + j], (*resImg)(i,j));
//             }
//         }
//     }
//     MatrixXd copyRes = *resImg;
//     for(int i = fmax(min_y,0); i < max_y; i++){
//         for(int j = fmax(min_x,0); j < max_x; j++){
//             if((*resImg)(i, j) == 0){
//                 if (i+1 < max_y && copyRes(i+1,j) != 0){ 
//                     (*resImg)(i, j) = copyRes(i+1,j);
//                 }else if(i-1 >= fmax(min_y,0) && copyRes(i-1,j) != 0){
//                     (*resImg)(i, j) = copyRes(i-1,j);
//                 }else if(j+1 < max_x && copyRes(i,j+1) != 0){
//                     (*resImg)(i, j) = copyRes(i,j+1);
//                 }else if(j-1 >=fmax(min_x,0) && copyRes(i,j-1) != 0){
//                     (*resImg)(i,j) = copyRes(i,j-1);
//                 }else if(i+1 < max_y && j+1 < max_x && copyRes(i+1,j+1)){
//                     (*resImg)(i,j) = copyRes(i+1,j+1);
//                 }else if(i-1 >= fmax(min_y,0) && j+1 < max_x && copyRes(i-1,j+1)){
//                     (*resImg)(i,j) = copyRes(i-1,j+1);
//                 }else if(i+1 < max_y && j-1 >=fmax(min_x,0) && copyRes(i+1,j-1)){
//                     (*resImg)(i,j) = copyRes(i+1,j-1);
//                 }else if(i-1 >= fmax(min_y,0) && j-1 >=fmax(min_x,0) && copyRes(i-1,j-1)){
//                     (*resImg)(i,j) = copyRes(i-1,j-1);
//                 }
//             }
//         }
//     }
//     // return resImg;
// }

//newer one
void placeImage(std::vector<unsigned char*> newImRs, std::vector<unsigned char*> newImGs, std::vector<unsigned char*> newImBs, std::vector<int> newImWidths, std::vector<int> newImHeights, MatrixXd* resImageR, 
            MatrixXd* resImageG, MatrixXd* resImageB, std::vector<double> min_xs, std::vector<double> min_ys, std::vector<double> max_xs, std::vector<double> max_ys){
// void placeImage(unsigned char* newImR, unsigned char* newImG, unsigned char* newImB, int newImWidth, int newImHeight, MatrixXd* resImageR, 
//             MatrixXd* resImageG, MatrixXd* resImageB, double min_x, double min_y, double max_x, double max_y){
// void placeImage(unsigned char* newImage, int newImWidth, int newImHeight, MatrixXd* resImg, double min_x, double min_y, double max_x, double max_y){
    // int w = newImage.cols();
    // int h = newImage.rows();
    // printf("w: %d, h: %d", w, h);
    std::cout << "Entered place image" << std::endl;
    double startTime = CycleTimer::currentSeconds();
    // int start_i = (int)fmax(min_y,0);
    // int start_j = (int)fmax(min_x,0);
    // int imNum = 0;
    // #pragma omp parallel for collapse(2) 
    // int imNum = 0;
    #pragma omp parallel for collapse(4) //schedule(dynamic)
    for(int imNum=0; imNum<newImRs.size(); imNum++){
        for (int i = (int)fmax(min_ys[imNum],0); i < (int)max_ys[imNum]; i++){ //access as row col
            for (int j = (int)fmax(min_xs[imNum],0); j < (int)max_xs[imNum]; j++){
                for(int im = 0; im<3; im++){
                    if(im == 0){
                        if ((*resImageR)(i,j) == 0){
                            (*resImageR)(i,j) = newImRs[imNum][i*newImWidths[imNum] + j]; //(i,j);
                        }
                        if ((*resImageR)(i,j) != 0 && newImRs[imNum][i*newImWidths[imNum] + j] != 0){
                            (*resImageR)(i,j) = fmax(newImRs[imNum][i*newImWidths[imNum] + j], (*resImageR)(i,j));
                        }
                    }else if(im == 1){
                        if ((*resImageG)(i,j) == 0){
                            (*resImageG)(i,j) = newImGs[imNum][i*newImWidths[imNum] + j]; //(i,j);
                        }
                        if ((*resImageG)(i,j) != 0 && newImGs[imNum][i*newImWidths[imNum] + j] != 0){
                            (*resImageG)(i,j) = fmax(newImGs[imNum][i*newImWidths[imNum] + j], (*resImageG)(i,j));
                        }
                    }else if(im == 2){
                        if ((*resImageB)(i,j) == 0){
                            (*resImageB)(i,j) = newImBs[imNum][i*newImWidths[imNum] + j]; //(i,j);
                        }
                        if ((*resImageB)(i,j) != 0 && newImBs[imNum][i*newImWidths[imNum] + j] != 0){
                            (*resImageB)(i,j) = fmax(newImBs[imNum][i*newImWidths[imNum] + j], (*resImageB)(i,j));
                        }
                    }
                }
            }
        }
    }
    std::cout << "Finished first loop place image" << std::endl;
    MatrixXd copyResR = (*resImageR);
    MatrixXd copyResG = (*resImageG);
    MatrixXd copyResB = (*resImageB);
    #pragma omp parallel for collapse(4) 
    for(int imNum=0; imNum<newImRs.size(); imNum++){
    // #pragma omp parallel for collapse(2) schedule(dynamic)
        for(int i = fmax(min_ys[imNum],0); i < (int)max_ys[imNum]; i++){
            for(int j = fmax(min_xs[imNum],0); j < (int)max_xs[imNum]; j++){
                for(int im = 0; im<3; im++){
                    int max_y = max_ys[imNum];
                    int min_y = fmax(min_ys[imNum],0);
                    int max_x = max_xs[imNum];
                    int min_x = fmax(min_xs[imNum],0);
                    if(im == 0 && (*resImageR)(i, j) == 0){
                        if (i+1 < max_y && copyResR(i+1,j) != 0){ 
                            (*resImageR)(i, j) = copyResR(i+1,j);
                        }else if(i-1 >= fmax(min_y,0) && copyResR(i-1,j) != 0){
                            (*resImageR)(i, j) = copyResR(i-1,j);
                        }else if(j+1 < max_x && copyResR(i,j+1) != 0){
                            (*resImageR)(i, j) = copyResR(i,j+1);
                        }else if(j-1 >=fmax(min_x,0) && copyResR(i,j-1) != 0){
                            (*resImageR)(i,j) = copyResR(i,j-1);
                        }else if(i+1 < max_y && j+1 < max_x && copyResR(i+1,j+1)){
                            (*resImageR)(i,j) = copyResR(i+1,j+1);
                        }else if(i-1 >= fmax(min_y,0) && j+1 < max_x && copyResR(i-1,j+1)){
                            (*resImageR)(i,j) = copyResR(i-1,j+1);
                        }else if(i+1 < max_y && j-1 >=fmax(min_x,0) && copyResR(i+1,j-1)){
                            (*resImageR)(i,j) = copyResR(i+1,j-1);
                        }else if(i-1 >= fmax(min_y,0) && j-1 >=fmax(min_x,0) && copyResR(i-1,j-1)){
                            (*resImageR)(i,j) = copyResR(i-1,j-1);
                        }
                    }if(im == 1 && (*resImageG)(i, j) == 0){
                        if (i+1 < max_y && copyResG(i+1,j) != 0){ 
                            (*resImageG)(i, j) = copyResG(i+1,j);
                        }else if(i-1 >= fmax(min_y,0) && copyResG(i-1,j) != 0){
                            (*resImageG)(i, j) = copyResG(i-1,j);
                        }else if(j+1 < max_x && copyResG(i,j+1) != 0){
                            (*resImageG)(i, j) = copyResG(i,j+1);
                        }else if(j-1 >=fmax(min_x,0) && copyResG(i,j-1) != 0){
                            (*resImageG)(i,j) = copyResG(i,j-1);
                        }else if(i+1 < max_y && j+1 < max_x && copyResG(i+1,j+1)){
                            (*resImageG)(i,j) = copyResG(i+1,j+1);
                        }else if(i-1 >= fmax(min_y,0) && j+1 < max_x && copyResG(i-1,j+1)){
                            (*resImageG)(i,j) = copyResG(i-1,j+1);
                        }else if(i+1 < max_y && j-1 >=fmax(min_x,0) && copyResG(i+1,j-1)){
                            (*resImageG)(i,j) = copyResG(i+1,j-1);
                        }else if(i-1 >= fmax(min_y,0) && j-1 >=fmax(min_x,0) && copyResG(i-1,j-1)){
                            (*resImageG)(i,j) = copyResG(i-1,j-1);
                        }
                    }else if(im == 2 && (*resImageB)(i, j) == 0){
                        if (i+1 < max_y && copyResB(i+1,j) != 0){ 
                            (*resImageB)(i, j) = copyResB(i+1,j);
                        }else if(i-1 >= fmax(min_y,0) && copyResB(i-1,j) != 0){
                            (*resImageB)(i, j) = copyResB(i-1,j);
                        }else if(j+1 < max_x && copyResB(i,j+1) != 0){
                            (*resImageB)(i, j) = copyResB(i,j+1);
                        }else if(j-1 >=fmax(min_x,0) && copyResB(i,j-1) != 0){
                            (*resImageB)(i,j) = copyResB(i,j-1);
                        }else if(i+1 < max_y && j+1 < max_x && copyResB(i+1,j+1)){
                            (*resImageB)(i,j) = copyResB(i+1,j+1);
                        }else if(i-1 >= fmax(min_y,0) && j+1 < max_x && copyResB(i-1,j+1)){
                            (*resImageB)(i,j) = copyResB(i-1,j+1);
                        }else if(i+1 < max_y && j-1 >=fmax(min_x,0) && copyResB(i+1,j-1)){
                            (*resImageB)(i,j) = copyResB(i+1,j-1);
                        }else if(i-1 >= fmax(min_y,0) && j-1 >=fmax(min_x,0) && copyResB(i-1,j-1)){
                            (*resImageB)(i,j) = copyResB(i-1,j-1);
                        }
                    }
                }
            }
        }
    }
    double endTime = CycleTimer::currentSeconds();
    std::cout << "Place image time " << endTime-startTime << std::endl;
}


void write_pgm(const char *filename, unsigned char *data, int w, int h)
{
    FILE *out_file;
    assert(w > 0);
    assert(h > 0);

    out_file = fopen(filename, "wb");
    if (!out_file) {
        fprintf(stderr, "Fail to open file: %s\n", filename);
        return;
    }

    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n", w, h);
    fwrite(data, sizeof(unsigned char), w * h, out_file);
    fclose(out_file);
}


int main(int argc, char *argv[])
{

    if (argc < 3) {
        printf("Please input at least two image filenames.\n");
        printf("usage: image_match img1 img2 ...\n");
        return -1;
    }
    double startTime = CycleTimer::currentSeconds();
    
    std::vector<ezsift::Image<unsigned char> > images;
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<unsigned char*> png_images;
    std::vector<unsigned char*> png_alpha;
    std::vector<unsigned char*> png_r;
    std::vector<unsigned char*> png_g;
    std::vector<unsigned char*> png_b;
    std::vector<char * > files; //Should probably switch away from this when switching to video
   
    // All image files
    for(int i=1; i<argc; i++){
        char* file = (char *)calloc(sizeof(char), strlen(argv[i]));
        memcpy(file, argv[i], sizeof(char) * strlen(argv[i]));
        file[strlen(argv[i])] = 0;
        files.push_back(file);
        ezsift::Image<unsigned char> image;

        //Finally can convert pngs
        
        std::vector<unsigned char> img;
        unsigned width, height;
        unsigned error = lodepng::decode(img, width, height, file);
        if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        
        unsigned char* data = img.data();
        unsigned char* new_data = new unsigned char[width * height];
        unsigned char* r = new unsigned char[width * height];
        unsigned char* g = new unsigned char[width * height];
        unsigned char* b = new unsigned char[width * height];
        unsigned char* a = new unsigned char[width * height];
        int j; 

        for( j=0; j< width*height; j++){
            new_data[j] = data[4*j]/3 + data[4*j+1]/3 + data[4*j+2]/3;
            r[j] = data[4*j];
            g[j] = data[4*j+1];
            b[j] = data[4*j+2];
            a[j] = data[4*j+3];
        }
        write_pgm("tmp.pgm", new_data, width, height);
        png_images.push_back(new_data);
        widths.push_back(width);
        heights.push_back(height);
        png_r.push_back(r);
        png_b.push_back(b);
        png_g.push_back(g);
        png_alpha.push_back(a);

        if (image.read_pgm("tmp.pgm") != 0) {
            std::cerr << "Failed to open input image!" << std::endl;
            return -1;
        }
        images.push_back(image);
    }
    double readingImagesEnd = CycleTimer::currentSeconds();
    std::cout << "Reading images time: " << readingImagesEnd-startTime << std::endl;
    
    //Parallel
    
    std::vector<std::list<ezsift::SiftKeypoint>> kpt_lists;
    for(int i=0; i<images.size(); i++){
        std::list<ezsift::SiftKeypoint> kpt_list;
        kpt_lists.push_back(kpt_list); //empty kpt_lists
    }
    
    ezsift::double_original_image(true);
    double siftStart = CycleTimer::currentSeconds();

    #pragma omp parallel for schedule(dynamic) 
    for(int i=0; i<images.size()+1; i++){
        if(i < images.size()){
            sift_cpu(images[i], kpt_lists[i], true);
        }else{
            dummyWarmup();
        }
    }
    // ezsift::sift_gpu(images, kpt_lists, true);
    double siftEnd = CycleTimer::currentSeconds();
    std::cout << "Sift time: " << siftEnd-siftStart << std::endl;
    std::vector<std::list<ezsift::MatchPair>> matches(images.size()-1);

    double findMatchesStart = CycleTimer::currentSeconds();

    bool matchListSizeZero = false;
    #pragma omp parallel for schedule(dynamic) //NEW REMOVAL
    for(int i=0; i<images.size()-1; i++){

        std::list<ezsift::MatchPair> match_list;
        // double matchKeyPointsStart = CycleTimer::currentSeconds();
        ezsift::match_keypoints(kpt_lists[i], kpt_lists[i+1], match_list); //Doesn't take long
        // double matchKeyPointsEnd = CycleTimer::currentSeconds();
        // std::cout << "Actual matching of keypoints time: " << matchKeyPointsEnd-matchKeyPointsStart << std::endl;

        matches[i] = match_list;
        if(match_list.size() == 0){
            matchListSizeZero = true;
        }
    }

    if(matchListSizeZero){ 
        std::cerr << "Failed to find any matches between two adjacent images!" << std::endl;
        return -1;
    }

    double findMatchesEnd = CycleTimer::currentSeconds();
    std::cout << "Generating matches time: " << findMatchesEnd-findMatchesStart << std::endl;

    //homographies multiplication must be sequential but ransac does not need to be
    double ransacStart = CycleTimer::currentSeconds();
    std::vector<MatrixXd> homographies;
    MatrixXd first = MatrixXd::Identity(3, 3);
    homographies.push_back(first);
    for(int i=1; i<images.size(); i++){
        // double ransacInnerStart = CycleTimer::currentSeconds();
        MatrixXd bestH = computeRansac(matches[i-1]);
        // std::cout << "Inner ransac: " << CycleTimer::currentSeconds() - ransacInnerStart << std::endl;
        homographies.push_back(homographies[i-1]*bestH);
    }
    double ransacEnd = CycleTimer::currentSeconds();
    std::cout << "Ransac time: " << ransacEnd-ransacStart << std::endl;
    //Parallel
    double findingDimsStart = CycleTimer::currentSeconds();
    int pano_min_x = 0; 
    int pano_min_y = 0; 
    int pano_max_x = images[0].w; 
    int pano_max_y = images[0].h; 

    
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
    }
    double findingDimsEnd = CycleTimer::currentSeconds();
    std::cout << "Finding dims time: " << findingDimsEnd-findingDimsStart << std::endl;

    double imgCompositionStart = CycleTimer::currentSeconds();
    int pan_height  = (int)(pano_max_y - pano_min_y); 
    int pan_width = (int)(pano_max_x - pano_min_x);

    MatrixXd resImageR = MatrixXd::Constant(pan_height, pan_width, 0);
    MatrixXd resImageG = MatrixXd::Constant(pan_height, pan_width, 0);
    MatrixXd resImageB = MatrixXd::Constant(pan_height, pan_width, 0);
    MatrixXd resImageA = MatrixXd::Constant(pan_height, pan_width, 0);
    
    std::vector<unsigned char*> newImageRs;
    std::vector<unsigned char*> newImageGs;
    std::vector<unsigned char*> newImageBs;
    std::vector<int> curr_widths;
    std::vector<int> curr_heights;
    std::vector<double> min_xs;
    std::vector<double> min_ys;
    std::vector<double> max_xs;
    std::vector<double> max_ys;
    // #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < images.size(); i++){
        double min_x; 
        double min_y; 
        double max_x; 
        double max_y; 
        findDimensions(images[i], homographies[i], &min_x, &min_y, &max_x, &max_y);      
        min_xs.push_back(min_x);
        min_ys.push_back(min_y);
        max_xs.push_back(max_x);
        max_ys.push_back(max_y);

        int curr_width = (int)(fmax(pano_max_x, max_x) - fmax(fmin(pano_min_x, min_x),0));
        int curr_height  = (int)(fmax(pano_max_y, max_y) - fmax(fmin(pano_min_y, min_y),0)); 

        //does not cache that well-- look into this
        unsigned char* newImR = new unsigned char[curr_height*curr_width]{};
        unsigned char* newImG = new unsigned char[curr_height*curr_width]{};
        unsigned char* newImB = new unsigned char[curr_height*curr_width]{};
        // unsigned char* newImA = new unsigned char[curr_height*curr_width]{255};
        curr_heights.push_back(curr_height);
        curr_widths.push_back(curr_width);

        double warpPerspectiveStart = CycleTimer::currentSeconds();
        warpPerspective(png_r[i], png_g[i], png_b[i], widths[i], heights[i], newImR, newImG, newImB, homographies[i], curr_width, curr_height);
        // warpPerspective(png_r[i], png_g[i], png_b[i], png_alpha[i], widths[i], heights[i], &newImR, &newImG, &newImB, &newImA, homographies[i]);
        double warpPerspectiveEnd = CycleTimer::currentSeconds();
        std::cout << "Warp perspective time: " << warpPerspectiveEnd-warpPerspectiveStart << std::endl;
        newImageRs.push_back(newImR);
        newImageGs.push_back(newImG);
        newImageBs.push_back(newImG);
    }
    // for()

    placeImage(newImageRs, newImageGs, newImageBs, curr_widths, curr_heights, &resImageR, &resImageG, &resImageB, min_xs, min_ys, max_xs, max_ys);
    // }
    double imgCompositionEnd = CycleTimer::currentSeconds();
    std::cout << "Img composition time: " << imgCompositionEnd-imgCompositionStart << std::endl;

    std::vector<unsigned char> resImg_vect;
    double finalLoopStart = CycleTimer::currentSeconds();
  
    for(int i=0; i<pan_height; i++){
        for(int j=0; j<pan_width; j++){
            resImg_vect.push_back(resImageR(i, j)); //color
            resImg_vect.push_back(resImageG(i, j));
            resImg_vect.push_back(resImageB(i, j));
            resImg_vect.push_back(255);
        }
    }
    double finalLoopEnd = CycleTimer::currentSeconds();
    std::cout << "Final loop time: " << finalLoopEnd-startTime << std::endl;

    unsigned err = lodepng::encode("result.png", resImg_vect, pan_width, pan_height);
    if(err) std::cout << "encoder error " << err << ": "<< lodepng_error_text(err) << std::endl;
    double endTime = CycleTimer::currentSeconds();

    std::cout << "Overall time: " << endTime-startTime << std::endl;
    return 0;
}
