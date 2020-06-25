#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

// Compute c = a + b.
static const char source[] =
"#if defined(cl_khr_fp64)\n"
"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#elif defined(cl_amd_fp64)\n"
"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#else\n"
"#  error double precision is not supported\n"
"#endif\n"
"kernel void mulmat(\n"
"       ulong n,\n"
"       global const int *a,\n"
"       global const int *b,\n"
"       global int *c\n,"
"       ulong    i\n"	
"       )\n"
"{\n"
"    size_t id = get_global_id(0);\n"
"    size_t j  = id/n;"
"    size_t k  = id%n;"
"    c[k*n+j] += a[k*n+i]*b[i*n+j];"
"}\n";

int main() {
	const size_t N = 2000;

	try {
		// Get list of OpenCL platforms.
		std::vector<cl::Platform> platform;
		cl::Platform::get(&platform);

		if (platform.empty()) {
			std::cerr << "OpenCL platforms not found." << std::endl;
			return 1;
		}

		// Get first available GPU device which supports double precision.
		cl::Context context;
		std::vector<cl::Device> device;
		for (auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
			std::vector<cl::Device> pldev;

			try {
				p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

				for (auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
					if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

					std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

					if (
						ext.find("cl_khr_fp64") == std::string::npos &&
						ext.find("cl_amd_fp64") == std::string::npos
						) continue;

					device.push_back(*d);
					context = cl::Context(device);
				}
			}
			catch (...) {
				device.clear();
			}
		}

		if (device.empty()) {
			std::cerr << "GPUs with double precision not found." << std::endl;
			return 1;
		}

		std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

		// Create command queue.
		cl::CommandQueue queue(context, device[0]);

		// Compile OpenCL program for found device.
		cl::Program program(context, cl::Program::Sources(
			1, std::make_pair(source, strlen(source))
		));

		try {
			program.build(device);
		}
		catch (const cl::Error&) {
			std::cerr
				<< "OpenCL compilation error" << std::endl
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
				<< std::endl;
			return 1;
		}

		cl::Kernel mulmat(program, "mulmat");

		// Prepare input data.
		std::vector<int> a(N*N);
		std::vector<int> b(N*N);
		std::vector<int> c(N*N, 0);

		for (int i = 0; i < N * N; ++i) {
			a[i] = rand() % 10;
			b[i] = rand() % 10;
		}

		/*for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				std::cout << a[i * N + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << " -------------------------------- " << std::endl;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				std::cout << b[i * N + j] << " ";
			}
			std::cout << std::endl;
		}*/
		//std::cout << " -------------------------------- " << std::endl;
		// Allocate device buffers and transfer input data to device.
		cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			a.size() * sizeof(int), a.data());

		cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			b.size() * sizeof(int), b.data());

		cl::Buffer C(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			c.size() * sizeof(int), c.data());


		auto start = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < N; ++i) {
			// Set kernel parameters.
			mulmat.setArg(0, static_cast<cl_ulong>(N));
			mulmat.setArg(1, A);
			mulmat.setArg(2, B);
			mulmat.setArg(3, C);
			mulmat.setArg(4, static_cast<cl_ulong>(i));

			// Launch kernel on the compute device.
			queue.enqueueNDRangeKernel(mulmat, cl::NullRange, N*N, cl::NullRange);
		}

		auto finish = std::chrono::high_resolution_clock::now();

		// Get result back to host.
		queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(int), c.data());

		std::chrono::duration<double> elapsed = finish - start;

		/*for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				std::cout << c[i * N + j] << " ";
			}
			std::cout << std::endl;
		}*/
		std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	}
	catch (const cl::Error& err) {
		std::cerr
			<< "OpenCL error: "
			<< err.what() << "(" << err.err() << ")"
			<< std::endl;
		return 1;
	}
}
