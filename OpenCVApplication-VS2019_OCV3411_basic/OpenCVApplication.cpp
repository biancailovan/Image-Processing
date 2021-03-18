// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}


/* Lab 1*/
void grayScaleAdditive()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		//float k = 2;
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				//uchar neg = MAX_PATH - val;
				//dst.at<uchar>(i, j) = neg;
				float k = 2;
				float val_f = val + k;
				if (val_f > 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = val_f;
				}
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("grey scale image", dst);
		waitKey();
	}
}

void grayScaleMultiplication()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		//float k = 2;
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				//uchar neg = MAX_PATH - val;
				//dst.at<uchar>(i, j) = neg;
				float k = 2;
				float val_f = val * k;
				if (val_f > 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = val_f;
				}
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("grey scale image", dst);

		//imwrite("Images/kids.bmp", dst); //writes the destination to file
		waitKey();
	}
}

void splitIn4Colors()
{

	int height = 255;
	int width = 255;

	Mat img = Mat(height, width, CV_8UC3);

	//cadranul 1
	for (int i = 0; i < height / 2; i++){
		for (int j = 0; j < width / 2; j++){
			Vec3b v3 = { 255, 255, 255 };
			img.at<Vec3b>(i, j) = v3;
		}
	}

	//cadranul 2
	for (int i = 0; i < height / 2; i++){
		for (int j = width / 2; j < width; j++){
			Vec3b v3 = {0, 0, 255 };
			img.at<Vec3b>(i, j) = v3;
		}
	}

	//cadranul 3
	for (int i = height / 2; i < height; i++){
		for (int j = 0; j < width / 2; j++){
			Vec3b v3 = { 0, 255, 0 };
			img.at<Vec3b>(i, j) = v3;
		}
	}

	//cadranul 4
	for (int i = height / 2; i < height; i++){
		for (int j = width / 2; j < width; j++){
			Vec3b v3 = { 0, 255, 255 };
			img.at<Vec3b>(i, j) = v3;
		}
	}

	imshow("4 colors", img);
	waitKey();

}

void inversaMatricii() {
	float v[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.f, -1.0f, -2.0f, -3.3f, -4.0f };

	Mat mat1(3, 3, CV_32FC1, v);
	std::cout << mat1 << "\n\n";
	Mat mat2;
	mat2 = mat1.inv();
	std::cout << mat2 << '\n';

	waitKey(30000);
}

/* Lab 2 */
void RGB24() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		Mat channels[3];
		split(src, channels);
		Mat R = channels[0];
		Mat G = channels[1];
		Mat B = channels[2];
		
		imshow("Red", R);
		imshow("Green", G);
		imshow("Blue", B);
		waitKey(0);
	}
}

void colorToGrayScale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst.at<unsigned char>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;
			}
		}

		imshow("Src", src);
		imshow("Gray image", dst);
		waitKey(0);
	}
}

void GrayScaleToBinary() {
	char fname[MAX_PATH];
	
	while (openFileDlg(fname))
	{
		int threshold;
		std::cout << "Threshold = ";
		std::cin >> threshold;
		threshold %= 255;
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				//dst.at<unsigned char>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;
				if (src.at<uchar>(i, j) < threshold)
					dst.at<uchar>(i, j) = 0;
				if (src.at<uchar>(i, j) >= threshold)
					dst.at<uchar>(i, j) = 255;
			}
		}

		imshow("Src", src);
		imshow("Gray image", dst);
		waitKey(0);
	}
}


void RGBToHSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		Mat H_norm = Mat(height, width, CV_8UC1);
		Mat S_norm = Mat(height, width, CV_8UC1);
		Mat V_norm = Mat(height, width, CV_8UC1);
		Mat channels[3];

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				float b = (float)src.at<Vec3b>(i,j)[0] / 255.0f;
				float g = (float)src.at<Vec3b>(i, j)[1] / 255.0f;
				float r = (float)src.at<Vec3b>(i, j)[2] / 255.0f;

				float M = max(max(r,g),b);
				float m = min(min(r,g),b);

				float C = M - m;

				//value
				float V = M;
				float S, H;
				//saturarion
				if (V!=0) {
					S = C / V;
				}
				else { //negru
					S = 0;
				}

				//hue
				if (C != 0) {
					if (M == r) H = 60 * (g - b) / C;
					if (M == g) H = 120 + 60 * (b - r) / C;
					if (M == b) H = 240 + 60 * (r - g) / C;
				}
				else { //grayscale
					H = 0;
				}

				if (H < 0) {
					H = H + 360;
				}

				H_norm.at<uchar>(i, j) = H * 255.0f / 360.0f;
				S_norm.at<uchar>(i, j) = S * 255.0f;
				V_norm.at<uchar>(i, j) = V * 255.0f;


			}
		}

		Mat hsv;
		cvtColor(src, hsv, CV_BGR2HSV);
		split(hsv, channels);

		imshow("input image", src);

		imshow("H", H_norm);
		imshow("S", S_norm);
		imshow("V", V_norm);

		imshow("H0", channels[0]*255/180);
		imshow("S0", channels[1]);
		imshow("V0", channels[2]);
		waitKey(0);
	}
}

void isInside1() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		int i, j;
		std:: cout << "i=";
		std::cin >> i;
		std::cout << "j=";
		std::cin >> j;

		if (i > 0 && j > 0 && i < height && j < width) {
			printf("Is inside");
		}
		else {
			printf("Is not inside");
		}
	}
	waitKey(0);
}

int isInside(Mat img, int i, int j) {

	if (i >= 0 && j >= 0 && i < img.rows && j < img.cols)
		return 1;
	else
		return 0;
}

/* Lab 3 */
void showHistogram(const string& name, int* hist, const int hist_cols,

	const int hist_height) {

	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];

	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
		// colored in magenta

	}
	imshow(name, imgHist);
}

void calculateHistogram() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int hist[256];
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < 256; i++) {
			hist[i] = 0;
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				hist[src.at<uchar>(i, j)]++;
			}
		}

		imshow("Src", src);
		showHistogram("Histogram", hist, 256, 200);
		waitKey(0);
	}
}

void calculateHist() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int hist[256];
		int M = height * width;
		float p[256];
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < 256; i++) {
			hist[i] = 0;
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				hist[src.at<uchar>(i, j)]++;
			}
		}

		for (int i = 0; i < 256; i++) {
			p[i] = (float)hist[i] / M;
			//std::cout << p[i] << " ";
		}

		imshow("Src", src);
		showHistogram("Histogram", hist, 256, 200);
		waitKey(0);
	}
}

void calculateFDP(Mat src, int hist[256], float p[256]) {
	char fname[MAX_PATH];

		double t = (double)getTickCount(); // Get the current time [s]
		int height = src.rows;
		int width = src.cols;
		int M = height * width;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < 256; i++) {
			hist[i] = 0; //h(g)
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				hist[src.at<uchar>(i, j)]++;
			}
		}

		for (int i = 0; i < 256; i++) {
			p[i] = (float)hist[i]/M;
		}

		waitKey(0);

}

void reducedAcc() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
	double t = (double)getTickCount(); // Get the current time [s]

	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	int height = src.rows;
	int width = src.cols;
	int hist[256] = { 0 };
	//int size = 255 / nrAcc;
	int nrAcc = 2;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			hist[src.at<uchar>(i, j) / nrAcc]++;
		}
	}

	imshow("Src", src);
	showHistogram("Histogram", hist, 256, 200);
	waitKey(0);
	}
}


int WH = 5;
float TH = 0.0003;

void praguriMultiple() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(src, src, Size(5, 5), 0.8, 0.8);
		int height = src.rows;
		int width = src.cols;
		int k = 0;
		int hist[256];
		int M = height * width;
		float p[256]; //vector tip float de dim 256
		float average;
		int v[256];
		Mat dst = src.clone();

		calculateFDP(src, hist, p);
		std::vector<uchar>list;
		list.push_back(0);

		for (int g = WH; g < 255 - WH; g++) {
			average = 0;
			bool isMaxim = true;
			for (int i = -WH; i < WH; i++) {
				average += p[g + i];
				if (p[g] < p[g + i]) {
					isMaxim = false; //max local
				}
			}
			average = (float)average / (2 * WH + 1);
			if (p[g] > (average + TH) && isMaxim) {
				//inserare g => lista de maxime
				list.push_back(g);
			}
			//}
		}

		list.push_back(255);
		//showHistogram("new histo", hist, 256, 200, list);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 0; k < list.size() - 1; k++) {
					uchar val = src.at<uchar>(i, j);
					if (val >= list.at(k) && val <= list.at(k + 1)) {
						if (val < (list.at(k) + list.at(k + 1)) / 2)
							dst.at<uchar>(i, j) = list.at(k);
						else
							dst.at<uchar>(i, j) = list.at(k + 1);

					}
				}
			}
		}
		
		imshow("Praguri multiple", dst);
		showHistogram("Histo", hist, 256, 200);
		waitKey(0);
	}
}

void floydSteinberg() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(src, src, Size(5, 5), 0.8, 0.8);
		int height = src.rows;
		int width = src.cols;
		int k = 0;
		int hist[256];
		int M = height * width;
		float p[256]; //vector tip float de dim 256
		float average;
		int v[256];
		Mat dst = src.clone();

		calculateFDP(src, hist, p);
		std::vector<uchar>list;
		list.push_back(0);

		for (int g = WH; g < 255 - WH; g++) {
			average = 0;
			bool isMaxim = true;
			for (int i = -WH; i < WH; i++) {
				average += p[g + i];
				if (p[g] < p[g + i]) {
					isMaxim = false; //max local
				}
			}
			average = (float)average / (2 * WH + 1);
			if (p[g] > (average + TH) && isMaxim) {
				//inserare g => lista de maxime
				list.push_back(g);
			}
			//}
		}

		list.push_back(255);
		showHistogram("Histo", hist, 256, 200);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 0; k < list.size() - 1; k++) {
					uchar val = src.at<uchar>(i, j);
					if (val >= list.at(k) && val <= list.at(k + 1)) {
						if (val < (list.at(k) + list.at(k + 1)) / 2)
							dst.at<uchar>(i, j) = list.at(k);
						else
							dst.at<uchar>(i, j) = list.at(k + 1);

					}
				}
			}
		}

		imshow("Praguri multiple", dst);
		//showHistogram("Histo", hist, 256, 200);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar new_pixel;
				uchar old_pixel = src.at<uchar>(i, j);
				for (int k = 0; k < list.size() - 1; k++) {
					//uchar val = src.at<uchar>(i, j);
					if (old_pixel >= list.at(k) && old_pixel <= list.at(k + 1)) {
						if (old_pixel < (list.at(k) + list.at(k + 1)) / 2)
							new_pixel = list.at(k);
						else
							new_pixel = list.at(k + 1);

					}
				}

				src.at<uchar>(i, j) = new_pixel;
				float error = old_pixel - new_pixel;
				if (isInside(src, i, j + 1)) {
					src.at<uchar>(i, j + 1) = src.at<uchar>(i, j + 1) + 7 * error / 16;
				}
				if (isInside(src, i + 1, j - 1)) {
					src.at<uchar>(i + 1, j - 1) = src.at<uchar>(i + 1, j - 1) + 3 * error / 16;
				}
				if (isInside(src, i + 1, j)) {
					src.at<uchar>(i + 1, j) = src.at<uchar>(i + 1, j) + 5 * error / 16;
				}
				if (isInside(src, i + 1, j + 1)) {
					src.at<uchar>(i + 1, j + 1) = src.at<uchar>(i + 1, j + 1) + error / 16;
				}

			}
		}
		imshow("Floyd-Steinberg", src);
		waitKey(0);		
	}
	
}

/* Lab 4 */

void func_I(Mat src, Mat mat){
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++){
			if (src.at<uchar>(i, j) != 0){
				mat.at<uchar>(i, j) = 1;
			}
			else{
				mat.at<uchar>(i, j) = 0;
			}
		}
}

bool equals(Vec3b pixel1, Vec3b pixel2) {
	return (pixel1[0] == pixel2[0]) && (pixel1[1] == pixel2[1]) && (pixel1[2] == pixel2[2]);
}

int area(Mat src, Vec3b pixel) {
	int area = 0;
	int width = src.cols;
	int height = src.rows;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (equals(pixel, src.at<Vec3b>(i, j))) {
				area += 1;
			}
		}
	}
	return area;
}

void center(Mat src, Vec3b pixel, int& ri, int& ci) {
	int ar = area(src, pixel);
	int width = src.cols;
	int height = src.rows;
	int r = 0, c = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (equals(pixel, src.at<Vec3b>(i, j))) {
				r = r + i;
				c = c + j;
			}
		}
	}
	ri = r / ar;
	ci = c / ar;
}

bool check_InCenter(Mat src, int i, int j, Vec3b pixel) {
	int width = src.cols;
	int height = src.rows;
	for (int m = i - 1; m <= i + 1; m++) {
		for (int n = j - 1; n <= j + 1; n++) {
			if (m >= 0 && m < height && n >= 0 && n < width && m != n) {
				if (!equals(src.at<Vec3b>(m, n), pixel)) {
					return false;
				}
			}
		}
	}
	return true;
}

int perimetru(Mat src, Vec3b pixel) {
	int perim = 0;
	int width = src.cols;
	int height = src.rows;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (equals(pixel, src.at<Vec3b>(i, j)) && !check_InCenter(src, i, j, pixel)) {
				perim = perim + 1;
			}
		}
	}

	return perim;
}


float axaAlungire(Mat src, Vec3b pixel, int ri, int ci) {
	int num = 0, n1 = 0, n2 = 0;
	int width = src.cols;
	int height = src.rows;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (equals(pixel, src.at<Vec3b>(i, j))) {
				num = num+ (i - ri) * (j - ci);
				n1 = n1 + (j - ci) * (j - ci);
				n2 = n2 + (i - ri) * (i - ri);
			}
		}
	}

	float teta = (atan2((2 * num),(n1-n2)))/ 2;
	float angle_grade = (teta * 180) / PI;
	return teta;
}

float get_thinness_ratio(int area, int perimeter) {
	return (4 * PI * area) / (perimeter*perimeter);
}

float elongation(Mat src, Vec3b pixel) {
	int cmax = 0, rmax = 0,cmin = src.rows, rmin = src.cols;
	int width = src.cols;
	int height = src.rows;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (equals(pixel, src.at<Vec3b>(i, j))) {
				if (i < rmin) { rmin = i; }
				if (i > rmax) { rmax = i; }
				if (j < cmin) { cmin = j; }
				if (j > cmax) { cmax = j; }
			}
		}
	}

	float aspect_ratio = (float)(cmax - cmin + 1) / (rmax - rmin + 1);
	return aspect_ratio;
}

void projections(Mat src, Vec3b p) {
	int height = src.rows;
	int width = src.cols;
	int* h = (int*)malloc(sizeof(int) * height);
	int* v = (int*)malloc(sizeof(int) * width);
	memset(h, 0, sizeof(int) * height);
	memset(v, 0, sizeof(int) * width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (equals(p, src.at<Vec3b>(i, j))) {
				h[i]++;
				v[j]++;
			}
		}
	}


	showHistogram("Orizontala", h, height, 200);
	showHistogram("Verticala", v, width, 200);
}

void DrawCross(Mat& img, Point p, int size, Scalar color, int thickness){
	line(img, Point(p.x - size / 2, p.y), Point(p.x + size / 2, p.y), color, thickness, 8);
	line(img, Point(p.x, p.y - size / 2), Point(p.x, p.y + size / 2), color, thickness, 8);
}


void onMouse(int event, int x, int y, int flags, void* param) {
	Mat* src = (Mat*)param;
	Mat copie = (*src).clone();
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
		Vec3b p = (*src).at<Vec3b>(y, x);
		int r, c;
		int a = area(*src, p);
		int perimeter = perimetru(*src, p);
		center(*src, p, r, c);
		float unghi_alungire = axaAlungire(*src, p, r, c);
		std::cout << "Aria = " << a << '\n';
		std::cout << "Centru masa: (" << c << ":" << r << ")\n";
		std::cout << "Perimetru = " << perimeter << '\n';
		std::cout << "Unghi axa alungire = " << unghi_alungire << '\n'; //teta
		std::cout << "Factor subtiere = " << get_thinness_ratio(a, perimeter) << '\n';
		std::cout << "Aspect ratio = " << elongation(*src, p) << '\n';

		projections(*src, p);
		int delta = 30;
		Point P1, P2;
		copie.at<Vec3b>(c, r) = Vec3b(0, 0, 0);
		//draw the center of mass
		DrawCross(copie, Point(c, r), 20, Scalar(255, 255, 255), 2);

		P1.x = c - delta;
		P1.y = r - (int)(delta * tan(axaAlungire(*src, p, r, c))); // teta is the elongation angle in radians
		P2.x = c + delta;
		float teta = ((axaAlungire(*src, p, r, c)));
		P2.y = r + (int)(delta * tan(teta));
		line(copie, P1, P2, Scalar(0, 0, 0), 1, 8);
		imshow("Copie", copie);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		//setMouseCallback("My Window", MyCallBackFunc, &src);
		setMouseCallback("My Window", onMouse, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Grey scale additive factor\n");
		printf(" 11 - Grey scale multiplicative factor\n");
		printf(" 12 - Split image in 4 colors\n");
		printf(" 13 - Inverse matrix\n");
		printf(" 14 - Color To Gray\n"); //2
		printf(" 15 - Gray scale to Binary\n"); //3
		printf(" 16 - Split 3 matrices\n"); //1
		printf(" 17 - RGB to HSV\n"); //4
		printf(" 18 - Is Inside\n"); //5
		printf(" 19 - Histogram\n");
		printf(" 20 - Reduced histo\n");
		printf(" 21 - Multilevel thresholding\n");
		printf(" 22 - Floyd-Steinberg dithering \n");
		printf(" 0 - Exit\n\n");

		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				//grayScaleFunction();
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				grayScaleAdditive();
				break;
			case 11:
				grayScaleMultiplication();
				break;
			case 12:
				splitIn4Colors();
				break;
			case 13:
				inversaMatricii();
				break;
			case 14:
				//colorToGrayScale();
				testColor2Gray();
				break;
			case 15:
				GrayScaleToBinary();
				break;
			case 16: 
				RGB24();
				break;
			case 17:
				RGBToHSV();
				break;
			case 18:
				isInside1();
				break;
			case 19:
				calculateHistogram();
				break;
			case 20:
				reducedAcc();
				break;
			case 21:
				praguriMultiple();
				break;
			case 22:
				floydSteinberg();
				break;
			
		}
	}
	while (op!=0);
	return 0;
}