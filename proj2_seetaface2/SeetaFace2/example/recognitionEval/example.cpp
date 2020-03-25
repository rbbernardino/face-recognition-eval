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

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

typedef std::chrono::duration<float> fsec;
const fs::path TRAIN_IMAGES_DIR = fs::path("./train_images");
const fs::path TEST_IMAGES_DIR = fs::path("./test_frames");

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
    // float threshold = 0.7f;
    float threshold = 0.5f;

    //set face detector's min face size
    engine.FD.set( seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 40 );

    
    std::vector<fs::path> trainImageVec;
    copy(fs::directory_iterator(TRAIN_IMAGES_DIR), fs::directory_iterator(), back_inserter(trainImageVec));
    // std::sort(trainImageVec.begin(), trainImageVec.end(), doj::alphanum_less<std::string>());
    std::sort(trainImageVec.begin(), trainImageVec.end());

    std::cerr << "Registering images... " << std::flush;
    auto startRegister = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> GalleryIndex( trainImageVec.size() );
    float ftimeRegisterAll = 0.0, ftimeReadAll = 0.0;
    for( size_t i = 0; i < trainImageVec.size(); ++i )
    {
	//register face into facedatabase
	std::string filename = trainImageVec[i].string();
	int64_t &index = GalleryIndex[i];

	// std::cerr << "Reading... " << filename << std::endl;
	// auto startRead = std::chrono::high_resolution_clock::now();
	seeta::cv::ImageData image = cv::imread( filename );
	// auto endRead = std::chrono::high_resolution_clock::now();
	// fsec timeReadImage = endRead-startRead;
	// std::cerr << "Time to read: " << timeReadImage.count() << "s" << std::endl;

	// std::cerr << "Registering... " << filename << std::endl;
	// auto startRegister = std::chrono::high_resolution_clock::now();
	auto id = engine.Register( image );
	// auto endRegister = std::chrono::high_resolution_clock::now();
	// fsec timeRegisterImage = endRegister-startRegister;
	// std::cerr << "Time to register: " << timeRegisterImage.count() << "s" << std::endl;

	index = id;
	// std::cerr << "Registered id = " << id << std::endl << std::endl;
	// ftimeRegisterAll += timeRegisterImage.count();
	// ftimeReadAll += timeReadImage.count();
    }
    auto endRegister = std::chrono::high_resolution_clock::now();
    fsec timeRegisterImages = endRegister-startRegister;
    std::cerr << "done! " << timeRegisterImages.count() << "s elapsed" << std::endl;

    std::map<int64_t, std::string> GalleryIndexMap;
    for( size_t i = 0; i < GalleryIndex.size(); ++i )
    {
        // save index and name pair
        if( GalleryIndex[i] < 0 ) continue;
        GalleryIndexMap.insert( std::make_pair( GalleryIndex[i], trainImageVec[i].stem().string() ) );
    }

    std::vector<fs::path> testImagesVec;
    copy(fs::directory_iterator(TEST_IMAGES_DIR), fs::directory_iterator(), back_inserter(testImagesVec));
    // sort(trainImageVec.begin(), trainImageVec.end(), doj::alphanum_less<std::string>());
    sort(trainImageVec.begin(), trainImageVec.end());

    for( size_t i = 0; i < testImagesVec.size(); ++i ) {
	cv::Mat frame = cv::imread(testImagesVec[i].string());
	seeta::cv::ImageData image = frame;

	// Detect all faces
	// std::cerr << "Detecting faces..." << std::endl;
	auto startDetection = std::chrono::high_resolution_clock::now();
	std::vector<SeetaFaceInfo> faces = engine.DetectFaces( image );
	auto endDetection = std::chrono::high_resolution_clock::now();
	fsec timeDetection = endDetection - startDetection;
	// std::cerr << "Time for detection: " << timeDetection.count() << std::endl;

	// std::cout << "Testing: " << testImagesVec[i].string() << " | " << faces.size() << " faces" << std::endl;
	if(faces.size() == 0)
	    std::cout << testImagesVec[i].stem().string() << " | " << "NO face detected!" << std::endl;
	int faceNumber = 1;
	for( SeetaFaceInfo &face : faces )
	{
	    // std::cerr << "Recognizing face " << faceNumber << std::endl;
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
		if (queried < 1) name = "unknown"; // no face queried from database
		else if( similarity > threshold ) // similarity greater than threshold, means recognized
		{
		    name = GalleryIndexMap[index];
		} else
		{
		    name = "unknown";
		}
	    }
	    auto endRecognition = std::chrono::high_resolution_clock::now();

	    std::cerr << testImagesVec[i].stem().string() << "               | " << "sim " << similarity << " | face " << faceNumber << " | ";
	    if(name.compare("ignored") == 0)
		std::cerr << "face has LOW quality!" << std::endl; 
	    else if(name.compare("unknown") == 0)
		std::cerr << "face NOT recognized!" << std::endl; 
	    else
		std::cerr << "DETECTED face: " << name << std::endl;
	    fsec timeRecognition = endRecognition - startRecognition;
	    // std::cerr << "Time for recognition, face " << faceNumber << ": " << timeRecognition.count() << "s" << std::endl;
	    faceNumber++;
	}
	// std::cout << std::endl;
    }
    return 0;
}
