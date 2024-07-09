hello: 
	nvcc -c matrix_blur_kernel.cu -o kernel.o
	g++ -c cv2_test.cpp -o  main.o `pkg-config --cflags --libs opencv4`
	g++ main.o kernel.o -o my_program -L/usr/local/cuda/lib64 -lcudart `pkg-config --cflags --libs opencv4`
