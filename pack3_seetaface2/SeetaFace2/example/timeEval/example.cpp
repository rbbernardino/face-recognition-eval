#pragma warning(disable: 4819)

#include <seeta/FaceEngine.h>

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <map>
#include <iostream>
#include <chrono> // measure time
#include <math.h> // pow

#include <seeta/QualityAssessor.h>

typedef std::chrono::duration<float> fsec;

int main(int argc, char *argv[])
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    int id = 0;
    seeta::ModelSetting FD_model( "./model/fd_2_00.dat", device, id );
    seeta::ModelSetting PD_model( "./model/pd_2_00_pts5.dat", device, id );
    seeta::ModelSetting FR_model( "./model/fr_2_10.dat", device, id );
    seeta::FaceEngine engine( FD_model, PD_model, FR_model, 4, 4 );

    seeta::QualityAssessor QA;

    // recognization threshold
    float threshold = 0.7f;

    //set face detector's min face size
    engine.FD.set( seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 80 );

    std::vector<std::string> GalleryImageFilename = { 
	"train_faces/benilton_12.jpg", "train_faces/benilton_7.jpg", "train_faces/cuevas_32.jpg", "train_faces/eduardo_36.jpg",
	"train_faces/benilton_13.jpg", "train_faces/cuevas_10.jpg", "train_faces/cuevas_39.jpg", "train_faces/eduardo_38.jpg",
	"train_faces/benilton_14.jpg", "train_faces/cuevas_14.jpg", "train_faces/cuevas_7.jpg", "train_faces/eduardo_4.jpg",
	"train_faces/benilton_32.jpg", "train_faces/cuevas_15.jpg", "train_faces/eduardo_11.jpg", "train_faces/eduardo_7.jpg",
	"train_faces/benilton_35.jpg", "train_faces/cuevas_19.jpg", "train_faces/eduardo_16.jpg",
	"train_faces/benilton_36.jpg", "train_faces/cuevas_3010.jpg", "train_faces/eduardo_28.jpg",
	"train_faces/benilton_6.jpg", "train_faces/cuevas_3019.jpg", "train_faces/eduardo_29.jpg"
    };
    // std::vector<std::string> GalleryImageFilename = { "img/train_1_benilton.jpg" };
    std::vector<int64_t> GalleryIndex( GalleryImageFilename.size() );
    float ftimeRegisterAll = 0.0, ftimeReadAll = 0.0;
    for( size_t i = 0; i < GalleryImageFilename.size(); ++i )
    {
        //register face into facedatabase
        std::string &filename = GalleryImageFilename[i];
        int64_t &index = GalleryIndex[i];

	std::cerr << "Reading... " << filename << std::endl;
	auto startRead = std::chrono::high_resolution_clock::now();
        seeta::cv::ImageData image = cv::imread( filename );
	auto endRead = std::chrono::high_resolution_clock::now();
	fsec timeReadImage = endRead-startRead;
	std::cerr << "Time to read: " << timeReadImage.count() << "s" << std::endl;

	std::cerr << "Registering... " << filename << std::endl;
	auto startRegister = std::chrono::high_resolution_clock::now();
        auto id = engine.Register( image );
	auto endRegister = std::chrono::high_resolution_clock::now();
	fsec timeRegisterImage = endRegister-startRegister;
	std::cerr << "Time to register: " << timeRegisterImage.count() << "s" << std::endl;

        index = id;
        std::cerr << "Registered id = " << id << std::endl << std::endl;
	ftimeRegisterAll += timeRegisterImage.count();
	ftimeReadAll += timeReadImage.count();
    }
    std::cerr << "Time to read all: " << ftimeReadAll << "s" << std::endl;
    std::cerr << "Time to register all: " << ftimeRegisterAll << "s" << std::endl;

    

    std::map<int64_t, std::string> GalleryIndexMap;
    for( size_t i = 0; i < GalleryIndex.size(); ++i )
    {
        // save index and name pair
        if( GalleryIndex[i] < 0 ) continue;
        GalleryIndexMap.insert( std::make_pair( GalleryIndex[i], GalleryImageFilename[i] ) );
    }
    
    // cv::Mat frame = cv::imread("img/teste_1_felipe.jpg");
    cv::Mat frame = cv::imread(argv[1]);
    
    seeta::cv::ImageData image = frame;

    // Detect all faces
    std::cerr << "Detecting faces..." << std::endl;
    auto startDetection = std::chrono::high_resolution_clock::now();
    std::vector<SeetaFaceInfo> faces = engine.DetectFaces( image );
    auto endDetection = std::chrono::high_resolution_clock::now();
    fsec timeDetection = endDetection - startDetection;
    std::cerr << "Time for detection: " << timeDetection.count() << std::endl;

    int faceNumber = 1;
    for( SeetaFaceInfo &face : faces )
    {
	std::cerr << "Recognizing face " << faceNumber << std::endl;
	auto startRecognition = std::chrono::high_resolution_clock::now();

	// Query top 1
	int64_t index = -1;
	float similarity = 0;

	auto points = engine.DetectPoints(image, face);
	std::string name;
	auto score = QA.evaluate(image, face.pos, points.data());
	if (score == 0) {
	    name = "ignored";
	} else {
	    auto queried = engine.QueryTop( image, points.data(), 1, &index, &similarity );
	    if (queried < 1) continue; // no face queried from database
	    if( similarity > threshold ) // similarity greater than threshold, means recognized
	    {
		name = GalleryIndexMap[index];
	    }
	}
	auto endRecognition = std::chrono::high_resolution_clock::now();

	if(name.empty())
	    std::cerr << "face NOT recognized!" << std::endl; 
	else
	    std::cout << "DETECTED face: " << name << std::endl;
	fsec timeRecognition = endRecognition - startRecognition;
	std::cerr << "Time for recognition, face " << faceNumber << ": " << timeRecognition.count() << "s" << std::endl;
    }
    return 0;
}
