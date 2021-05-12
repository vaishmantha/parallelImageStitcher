#include "ezsift.h"
#include "CycleTimer.h"
#include <limits>
#include <iostream>
#include <list>
#include <eigen/Eigen/Dense>

#include "lodepng/lodepng.h"

using Eigen::MatrixXd;

#define USE_FIX_FILENAME 0

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

    MatrixXd H = computeHomography(x1Norm(Eigen::seqN(0,x1Norm.rows()), {0, 1}), x2Norm(Eigen::seqN(0,x2Norm.rows()), {0, 1}));
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
        MatrixXd x1 = locs1(rand_inds, Eigen::seqN(0,locs1.cols())); 
        MatrixXd x2 = locs2(rand_inds, Eigen::seqN(0,locs2.cols())); 

        MatrixXd x1_res_h = homogeneous_loc1(rand_inds,  Eigen::seqN(0,homogeneous_loc1.cols())); 
        MatrixXd x2_res_h = homogeneous_loc2(rand_inds, Eigen::seqN(0,homogeneous_loc2.cols())); 

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
            diff = (prod.transpose()(i, {0,1})/prod.transpose()(i, 2) - locs1(i,  Eigen::seqN(0,locs1.cols()))).norm();
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
    MatrixXd x1_res = locs1(inlier_inds, Eigen::seqN(0,locs1.cols())); 
    MatrixXd x2_res = locs2(inlier_inds, Eigen::seqN(0,locs2.cols()));
    MatrixXd x1_res_h = homogeneous_loc1(inlier_inds, Eigen::seqN(0,homogeneous_loc1.cols())); 
    MatrixXd x2_res_h = homogeneous_loc2(inlier_inds, Eigen::seqN(0,homogeneous_loc2.cols()));
    MatrixXd bestNormalizedHomography = computeNormalizedHomography(x1_res, x2_res, x1_res_h, x2_res_h);
    // std::cout << bestNormalizedHomography << std::endl;
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
    MatrixXd lRow = imgDimss({2}, Eigen::seqN(0,imgDimss.cols())); 
    MatrixXd tm = lRow.replicate(3,1);
    for(int i=0; i< tm.rows(); i++){
        for(int j=0; j< tm.cols(); j++){
            tm(i,j) = 1.0/tm(i,j);
        }
    }
    imgDimss = imgDimss.cwiseProduct(tm);
    *min_x = imgDimss({0}, Eigen::seqN(0,imgDimss.cols())).minCoeff();
    *max_x = imgDimss({0}, Eigen::seqN(0,imgDimss.cols())).maxCoeff();
    *min_y = imgDimss({1}, Eigen::seqN(0,imgDimss.cols())).minCoeff();
    *max_y = imgDimss({1}, Eigen::seqN(0,imgDimss.cols())).maxCoeff();
}

MatrixXd warpPerspective(unsigned char* png_image, int png_width, int png_height, MatrixXd newIm, MatrixXd H){
    for(int i=0; i< png_height; i++){ 
        for(int j=0; j<png_width; j++){
            MatrixXd tmp = MatrixXd::Constant(1,3, 0.0);
            tmp(0,0) = j;
            tmp(0,1) = i;
            tmp(0,2) = 1;
            MatrixXd res = H*tmp.transpose();
            MatrixXd tm = res({2}, Eigen::seqN(0, res.cols())).replicate(3,1);
            res = res.cwiseQuotient(tm);
            if ((int)res(0,0) >= 0 && (int)res(0,0) < newIm.cols() && (int)res(1,0) >= 0 && (int)res(1,0) < newIm.rows()){
                newIm((int)res(1,0), (int)res(0,0)) = (int)png_image[i*png_width + j];
            }
            // else{
            //     newIm(i,j) = 255;
            // }
        }
    }
    return newIm;
}

MatrixXd placeImage(MatrixXd newImage, MatrixXd resImg){
    int w = newImage.cols();
    int h = newImage.rows();
    printf("w: %d, h: %d", w, h);
    // printf("base w: %d, h: %d", base.cols, base.rows);
    MatrixXd slice = resImg(Eigen::seqN(0,h), Eigen::seqN(0,w));
    // cv::Mat dstImg(base(cv::Rect(0, 0, w, h)));
    for (int i = 0; i < h; i++){ //access as row col
        for (int j = 0; j < w; j++){
            if (slice(i,j) == 0){
                // if(newImage(i, j) == 0 && i+1 < h && i-1 >=0 && j+1 < w && j-1 >=0 ){
                //     dstImg.at<uint8_t>(i, j) = 255; //(1/4)*(newImage(i+1,j)+newImage(i-1,j)+newImage(i,j+1)+newImage(i,j-1));
                // }else{
                // dstImg.at<uint8_t>(i, j) = newImage(i,j);
                slice(i,j) = newImage(i,j);
                // }
                 //newImage.at<uint8_t>(i, j);
            }
            if (slice(i,j) != 0 && newImage(i, j) != 0){
                //dstImg.at<uint8_t>(i, j) = fmax(newImage.at<uint8_t>(i, j), dstImg.at<uint8_t>(i, j));
                // dstImg.at<uint8_t>(i, j) = fmax(newImage(i,j), dstImg.at<uint8_t>(i, j));
                slice(i,j) = fmax(newImage(i,j), slice(i,j));
            }
        }
    }
    resImg(Eigen::seqN(0,h), Eigen::seqN(0,w)) = slice;
    return resImg;
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
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<unsigned char*> png_images;
    std::vector<unsigned char*> color_png_images;
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
            
            std::vector<unsigned char> img;
            unsigned width, height;
            unsigned error = lodepng::decode(img, width, height, file);
            if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
            
            unsigned char* data = img.data();
            unsigned char* new_data = new unsigned char[width * height];
            for(int i=0; i< width*height; i++){
                new_data[i] = data[4*i]/3 + data[4*i+1]/3 + data[4*i+2]/3;
            }
            write_pgm("tmp.pgm", new_data, width, height);
            png_images.push_back(new_data);
            widths.push_back(width);
            heights.push_back(height);
            color_png_images.push_back(data); //be careful

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
        // printf("pano_size: max y: %d, min_y: %d, max_x: %d, min_x: %d \n", pano_max_y, pano_min_y, pano_max_x,pano_min_x );
    }

    int pan_height  = (int)(pano_max_y - pano_min_y); 
    int pan_width = (int)(pano_max_x - pano_min_x);

    MatrixXd resImage = MatrixXd::Constant(pan_height, pan_width, 0);
    
    for (int i = 0; i < images.size(); i++){
        double min_x; 
        double min_y; 
        double max_x; 
        double max_y; 
        findDimensions(images[i], homographies[i], &min_x, &min_y, &max_x, &max_y); 
        max_x = fmax(pano_max_x, max_x); 
        max_y = fmax(pano_max_y, max_y); 
        min_x = fmax(fmin(pano_min_x, min_x),0); 
        min_y = fmax(fmin(pano_min_y, min_y),0);         

        int curr_width = (int)(max_x - min_x);
        int curr_height  = (int)(max_y - min_y); 

        MatrixXd newIm = MatrixXd::Constant(curr_height, curr_width, 0);
        newIm = warpPerspective(png_images[i], widths[i], heights[i], newIm, homographies[i]);

        resImage = placeImage(newIm, resImage);
        // std::cout << "After place image " << std::endl;
    }
    std::vector<unsigned char> resImg_vect;
    for(int i=0; i<pan_height; i++){
        for(int j=0; j<pan_width; j++){
            resImg_vect.push_back(resImage(i, j)); //color
            resImg_vect.push_back(resImage(i, j));
            resImg_vect.push_back(resImage(i, j));
            resImg_vect.push_back(255);
        }
    }
    
    unsigned err = lodepng::encode("result.png", resImg_vect, pan_width, pan_height);
    if(err) std::cout << "encoder error " << err << ": "<< lodepng_error_text(err) << std::endl;

    return 0;
}

