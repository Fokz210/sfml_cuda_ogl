#include <cmath>
#include <stdio.h>

#include <gl/glew.h>
#include <gl/GL.h>
#include <SFML/Graphics.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "color.h"
#include "sfml_context.h"
#include "Header.h"


constexpr auto M_PI = 3.14159265359;

__host__   void render (sfml_host_context<color> context, size_t frames);
__device__ __host__ color tranform_hue (const color & in, float H);

class hue_shifter
{
public:
	__host__ __device__ hue_shifter() :
		matrix(3, 3)
	{
		matrix = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	}

	__host__ __device__ void set_hue_rotation(float radians)
	{
		auto sinA = sinf(radians);
		auto cosA = cosf(radians);

		matrix[0][0] = cosA + (1.f - cosA) / 3.f;
		matrix[0][1] = 1.f / 3.f * (1.f - cosA) - sqrt(1.f / 3.f) * sinA;
		matrix[0][2] = 1.f / 3.f * (1.f - cosA) + sqrt(1.f / 3.f) * sinA;
		matrix[1][0] = 1.f / 3.f * (1.f - cosA) + sqrt(1.f / 3.f) * sinA;
		matrix[1][1] = cosA + 1.f / 3.f * (1.f - cosA);
		matrix[1][2] = 1.f / 3.f * (1.f - cosA) - sqrt(1.f / 3.f) * sinA;
		matrix[2][0] = 1.f / 3.f * (1.f - cosA) - sqrt(1.f / 3.f) * sinA;
		matrix[2][1] = 1.f / 3.f * (1.f - cosA) + sqrt(1.f / 3.f) * sinA;
		matrix[2][2] = cosA + 1.f / 3.f * (1.f - cosA);
	}

	__host__ __device__ color apply(const color & c)
	{
		auto rx = c.r * matrix[0][0] + c.g * matrix[0][1] + c.b * matrix[0][2];
		auto gx = c.r * matrix[1][0] + c.g * matrix[1][1] + c.b * matrix[1][2];
		auto bx = c.r * matrix[2][0] + c.g * matrix[2][1] + c.b * matrix[2][2];

		return color
		{
			static_cast<std::uint8_t> (clamp(rx)),
			static_cast<std::uint8_t> (clamp(gx)),
			static_cast<std::uint8_t> (clamp(bx)),
			c.a
		};
	}

protected:
	template<class T>
	__host__ __device__ T clamp(T val)
	{
		if (val < 0)
			return 0;
		
		if (val > 255)
			return 255;
	}

	matrix<float> matrix;
};


__global__ void render(sfml_device_context<color> context, size_t frames, hue_shifter shifter);

int main ()
{
	sf::RenderWindow window (sf::VideoMode (1920, 1080), "cuda + sfml");

	int device = 0;

	cudaFree (0);
	cudaGetDevice (&device);
	cudaGLSetGLDevice (device);

	printf_s ("%d", device);

	if (glewInit () != GLEW_OK)
		throw std::runtime_error ("glew fucked up");

	sfml_device_context<color> context (window.getSize().x, window.getSize().y);

	dim3 threads_per_block(16, 16);
	dim3 num_blocks(window.getSize().x / threads_per_block.x, window.getSize().y / threads_per_block.y);

	size_t frames = 0;

	hue_shifter shifter;

	while (window.isOpen ())
	{
		frames++;

		sf::Event event;

		while (window.pollEvent (event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		render <<< num_blocks, threads_per_block >>> (context, frames, shifter);

		//render (context, frames);

		context.update ();
		window.draw (context);
		context.clear ();
		window.display ();
		window.clear ();
	}
}

int mmain ()
{
	sf::RenderWindow window (sf::VideoMode (1920, 1080), "cuda + sfml");

	if (glewInit () != GLEW_OK)
	{
		printf ("fuck");
		return -1;
	}
	const auto size = 1920u * 1080u * 4u;

	int device = 0;

	cudaFree (0);
	cudaGetDevice (&device);
	cudaGLSetGLDevice (device);

	GLuint pbo{0};
	cudaGraphicsResource_t resource{0};

	void * device_ptr = nullptr;

	glGenBuffers (1, &pbo);

	glBindBuffer (GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData (GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_STREAM_DRAW);

	checkCudaCall (cudaGraphicsGLRegisterBuffer (&resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
}

__global__ void render (sfml_device_context<color> context, size_t frames, hue_shifter shifter)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	shifter.set_hue_rotation(static_cast<float> (frames) / 1000 * M_PI);

	auto gradient = color
	{
		static_cast<std::uint8_t>(static_cast<float>(x) / context.buffer.width() * 255.f),
		static_cast<std::uint8_t>(static_cast<float>(y) / context.buffer.height() * 255.f),
		255,
		255
	};

	context.buffer[y][x] = shifter.apply (gradient);
}

__host__ void render (sfml_host_context<color> context, size_t frames)
{
	for (auto y = 0u; y < context.buffer.height (); y++)
		for (auto x = 0u; x < context.buffer.width (); x++)
		{
			auto gradient = color
			{
				255,
				255,
				255,
				255
			};

			context.buffer[y][x] = tranform_hue (gradient, static_cast<float> (frames % 100) / 100.f);
		}
}

__device__ __host__ color tranform_hue (const color & in, float H)
{
	float U = cos (H * M_PI / 180);
	float W = sin (H * M_PI / 180);

	color ret;
	ret.r = (.299 + .701 * U + .168 * W) * in.r
		+ (.587 - .587 * U + .330 * W) * in.g
		+ (.114 - .114 * U - .497 * W) * in.b;
	ret.g = (.299 - .299 * U - .328 * W) * in.r
		+ (.587 + .413 * U + .035 * W) * in.g
		+ (.114 - .114 * U + .292 * W) * in.b;
	ret.b = (.299 - .3 * U + 1.25 * W) * in.r
		+ (.587 - .588 * U - 1.05 * W) * in.g
		+ (.114 + .886 * U - .203 * W) * in.b;
	ret.a = in.a;

	return ret;
}
