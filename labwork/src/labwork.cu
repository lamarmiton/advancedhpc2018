#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4
int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    Timer timerLab1; // We use another timer in lab1, so better to use another one here
    timer.start();
    switch (lwNum) {
        case 1:
	    timerLab1.start();
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timerLab1.getElapsedTimeInMilliSec());
            timerLab1.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
	    printf("labwork 1 OpenMP ellapsed %.1fms\n",lwNum, timerLab1.getElapsedTimeInMilliSec());
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma opm parallele for
    for (int j = 0; j < 100; j++) {             // let's do it 100 times, otherwise it's too fast!
	#pragma opm parallele for
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }

}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
  int nbDevices;	

  printf("Scanning devices ..\n");
  
  cudaGetDeviceCount(&nbDevices);	// Get the number of devices


  printf("We got %d devices here\n\n",nbDevices);

  for (int i = 0; i < nbDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device name: %s\n", prop.name);	// Display the name of the device
    printf("Device Number: %d\n", i);	// Display the id of the device
    printf("Number of core : %d\n",getSPcores(prop)); // Display number of core
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount); // Display the number of Multi processor
    printf("Warp Size : %d threads\n", prop.warpSize); // Display the wrapSize
    printf("Memory Clock Rate : %d kHz\n",
           prop.memoryClockRate);	// Display Memory ClockRate
    printf("Memory Bus Width : %d bits\n",
           prop.memoryBusWidth);	// Display Memory bus Width
    printf("Peak Memory Bandwidth : %f GB/s\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1e9);	//Display memory Brandwith
  }  
}

//Making the kernel

__global__ void grayscale(uchar3 *input, uchar3 *output) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
	output[tid].z = output[tid].y = output[tid].x;

}
	
void Labwork::labwork3_GPU() {
	
	// Get the basic variable such as the pixelcount block size etc ..
	int pixelCount = inputImage->width * inputImage->height;
	int blockSize = 1024;
	int numBlock = pixelCount / blockSize;

	uchar3 *devInput;
	uchar3 *devGray;
	
	// Initialize the output image
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	
	// Allocate the memory in the device for the Deviceinput and the Deviceouput
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
	
	// Copy from the HostInput to the devInput (here, the image)
	cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
	
	// Do the thing you want to do
	grayscale<<<numBlock, blockSize>>>(devInput, devGray);
	
	// Copy from the DeviceOutput to the HostOutput (here the image in grayscale)
	cudaMemcpy(outputImage, devGray,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
	
	// Don't forget to free
	cudaFree(devInput);
	cudaFree(devGray);
}

__global__ void grayscale2D(uchar3 *input, uchar3 *output, int imageWidth, int imageHeight) {

	// We need to know where we are rigth now
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	// Whecking if we are still in the image
	if(x>=imageWidth) return;
	if(y>=imageHeight) return;
	
	int tid = imageWidth * y + x; // RowSize * y + x

	// We turn the pixel gray
	output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
	output[tid].z = output[tid].y = output[tid].x;

}

void Labwork::labwork4_GPU() {
   	// Get the basic variable such as the pixelcount block size etc ..
	int pixelCount = inputImage->width * inputImage->height;
	dim3 blockSize = dim3(32, 32);
	dim3 gridSize = dim3(ceil(inputImage->width/blockSize.x), ceil(inputImage->height/blockSize.y));

	uchar3 *devInput;
	uchar3 *devGray;
	
	// Initialize the output image
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	
	// Allocate the memory in the device for the Deviceinput and the Deviceouput
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
	
	// Copy from the HostInput to the devInput (here, the image)
	cudaMemcpy(devInput, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice);
	
	// Do the thing you want to do
	grayscale2D<<<gridSize, blockSize>>>(devInput, devGray, inputImage->width, inputImage->height);
	
	// Copy from the DeviceOutput to the HostOutput (here the image in grayscale)
	cudaMemcpy(outputImage, devGray,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
	
	// Don't forget to free
	cudaFree(devInput);
	cudaFree(devGray);
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

void Labwork::labwork5_GPU() {
    
}

void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
